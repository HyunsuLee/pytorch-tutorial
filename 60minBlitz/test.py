import torch
import torch.nn as nn

conv = nn.Conv1d(25, 64, 1) #in_feature(channel), out_channel, kernel_size
para_list = list(conv.parameters())
#print(len(para_list))
print(para_list[0].size())

input = torch.rand(2, 25, 1) #n_batch, in_feature, time_step

out = conv(input)


print(out.size())


lin = nn.Linear(25, 64)
lin_input = torch.rand(2,25)
lin_para = list(lin.parameters())
print(lin_para[0].size())

lin_out = lin(lin_input)

print(lin_out.size())

# the reason of my confusion was from 
# the time_step concept in input data for the Conv1D.
# There's no time steps in Dense layer, the dimension of input for dense layer is
# just 1D. (except the batch size) But for Conv1d, there's additional dimension 
# as time_step. Thus, when see usecase conv1d be aware of time_step dimension of the inpu data.