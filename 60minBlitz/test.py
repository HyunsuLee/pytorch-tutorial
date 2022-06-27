import torch
import torch.nn as nn

conv = nn.Conv1d(25, 64, 1) #in_feature(channel), out_channel, kernel_size
para_list = list(conv.parameters())
#print(len(para_list))
print("Conv1d parameter = " + str(para_list[0].size()))

input = torch.rand(2, 25, 2) #n_batch, in_feature, time_step

out = conv(input)

print("input size = "+ str(input.size()))
print("Conv 1d out size = " + str(out.size()))


lin = nn.Linear(25, 64)

lin_input = input.transpose(1,2)

lin_para = list(lin.parameters())
print("linear parameter = " + str(lin_para[0].size()))
print("input size for linear= "+ str(lin_input.size()))
lin_out = lin(lin_input).transpose(1,2)

print("linear out size = " + str(lin_out.size()))

# the reason of my confusion was from 
# the time_step concept in input data for the Conv1D.
# There's no time steps in Dense layer, the dimension of input for dense layer is
# just 1D. (except the batch size) But for Conv1d, there's additional dimension 
# as time_step. Thus, when see usecase conv1d be aware of time_step dimension of the inpu data.