from ppo.ppo_datahelper import OnlyPromptDataset, PPOSFTDataset
import sys
from config import parse_args
from torch.utils.data import DataLoader


def OnlyPromptDataset_test(opt, accelerator):
    only_prompt_dataset = OnlyPromptDataset(opt, accelerator)
    prompt_loader = iter(DataLoader(
                                    OnlyPromptDataset(opt, accelerator, mode='train'), 
                                    batch_size=None, 
                                    num_workers=opt.num_workers, 
                                    prefetch_factor=opt.num_prefetch, 
                                    pin_memory=True))
    for one in next(prompt_loader):
        print(one)


def PPOSFTDataset_test(opt, accelerator):
    sft_dataset = PPOSFTDataset(opt=opt, accelerator=accelerator)
    pretrain_loader = iter(DataLoader(sft_dataset,
                                      batch_size=None, 
                                      num_workers=1,
                                      prefetch_factor=4,
                                      pin_memory=True))
    for one in next(pretrain_loader):
        print(one)


if __name__ == '__main__':
    opt = parse_args()
    accelerator = None
    OnlyPromptDataset_test(opt, accelerator=accelerator)

