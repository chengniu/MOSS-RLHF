from train_ppo import Llama, LlamaRewardModel
from ppo.ppo_datahelper import get_tokenizer
from config import parse_args
from utils import *
from ppo.ppo_datahelper import OnlyPromptDataset
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List


def strip_pad_token_id(seq: List[int], tokenizer):
        return [tok for tok in seq if tok != tokenizer.pad_token_id]


def concat_context_and_response(context: List[List[int]], responses: List[List[Tuple[float, List[int]]]], tokenizer):
        assert len(context) == len(responses), f'Size not match: {len(context)} and {len(responses)}'
        total_context, total_response = [], []
        for _context, _response in zip(context, responses):
            _context = strip_pad_token_id(_context, tokenizer)
            for _, resp in _response:
                resp = strip_pad_token_id(resp, tokenizer)
                if resp[-1] != tokenizer.eos_token_id:
                    logging.warn(f'Generated response is too long: {tokenizer.decode(_context + resp, skip_special_tokens=False)}')

                total_context.append(_context.copy())
                total_response.append(resp)

                # Debug
                # logging.info(f'===={self.tokenizer.decode(_context + resp, skip_special_tokens=False)}')
                
        total_gene_samples_vec = [c + r for c, r in zip(total_context, total_response)]
        return total_context, total_response, total_gene_samples_vec # total_context, total_response, total_gene_samples_vec


def make_experiences_test(opt, accelerator):
    tokenizer = get_tokenizer(opt)
    policy_model = Llama.from_pretrained(opt.policy_model_path, opt, tokenizer)
    reward_model = LlamaRewardModel.from_pretrained(opt.critic_model_path, opt, tokenizer)
    only_prompt_dataset = OnlyPromptDataset(opt, accelerator)
    prompt_loader = iter(DataLoader(
                                    OnlyPromptDataset(opt, accelerator, mode='train'), 
                                    batch_size=None, 
                                    num_workers=opt.num_workers, 
                                    prefetch_factor=opt.num_prefetch, 
                                    pin_memory=True))
    batch: Dict[str, Any] = next(prompt_loader)
    context_vec = batch['text_vec'].tolist()
    _, responses_vec = policy_model.generate(batch, to_half=False)
    assert len(context_vec) == len(responses_vec)
    context_vec_sampled, resp_vec_sampled, sampled_vec = concat_context_and_response(context_vec, responses_vec, tokenizer)
    sampled_vec = torch.tensor(pad_sequences(sampled_vec, pad_value=tokenizer.pad_token_id, padding='left'), 
                                             dtype=torch.long, device=None)
    rewards, *_ = reward_model(sampled_vec)


if __name__ == "__main__":
    opt = parse_args()
    make_experiences_test(opt, accelerator)
