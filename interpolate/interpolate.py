import torch
from torch.optim import SGD
import os

DATA_DIR = "/media/user11209/D/git_clone_file/nerfstudio/external/flexnerf/interpolate/test/test_data/test_data.pt"
SAVE_DIR = "/media/user11209/D/git_clone_file/nerfstudio/external/flexnerf/interpolate/test/pred_data"

# gaussian utils, each estimate_gaussian shape is [D, D+1], namely [sigma[D,D], mu[D,1]]
GAUSSIAN_N = 1
GAUSSIAN_D = 1
ESTIMATE_GAUSSIAN = torch.zeros([GAUSSIAN_N,GAUSSIAN_D,GAUSSIAN_D+1], device=torch.device("cuda:0"), requires_grad=True)
with torch.no_grad():
    for i in range(GAUSSIAN_D):
        ESTIMATE_GAUSSIAN[:,i,i] = 1
WEIGHT = torch.zeros([GAUSSIAN_N], device=torch.device("cuda:0"), requires_grad=True)
def get_sigma_mu_matrix(estimate_gaussian):
    '''
    input Tensor[B, D, D+1]
    given estimate_gaussian = (sigma, mu')
    ret   sigma_mu_matrix   =((sigma+mu'*mu, mu'),
                              (mu          , 1  ))
    '''
    B = estimate_gaussian.shape[0]
    D = estimate_gaussian.shape[1]
    sigma = estimate_gaussian[:, :D, :D]                                                    # [B,D,D]
    mu = estimate_gaussian[:, :D, -1:]                                                      # [B,D,1]
    last_row = torch.concat([torch.transpose(mu, 1, 2), 
                            torch.ones(B,1,1, dtype=mu.dtype, device=mu.device)], dim=2)    # [B,1,D+1]
    sigma_mu_matrix = torch.concat([estimate_gaussian, last_row], dim=1)                    # [B,D+1, D+1]
    mu_2_matrix = torch.matmul(mu, torch.transpose(mu,1,2))                                 # [B,D,D]
    sigma_mu_matrix[:, :D, :D] += mu_2_matrix
    return sigma_mu_matrix

def gaussian_before_train_hook(estimate_gaussian, weight, weight_decay=0.99):
    if estimate_gaussian.grad == None or weight.grad == None:
        assert estimate_gaussian.grad == None
        assert weight.grad == None
        return

    D = estimate_gaussian.shape[0]
    estimate_gaussian_var  = estimate_gaussian[:, :D, :D]              # [N, D, D]
    estimate_gaussian_mean = estimate_gaussian[:, :D, -1]              # [N, D]

    estimate_gaussian_square_grad = estimate_gaussian.grad[:, :D, :D] / weight.grad.view(-1, 1, 1)  # [N, D, D]  E(sample**2)
    estimate_gaussian_mean_grad = estimate_gaussian.grad[:, :D, -1]   / weight.grad.view(-1, 1)     # [N, D]     E(sample)

    with torch.no_grad():
        if weight == 0:
            weight[:] = weight.grad
        old_weight = weight_decay * weight / (weight_decay * weight + weight.grad)
        new_weight = weight.grad / (weight_decay * weight + weight.grad)

        new_estimate_gaussian_mean = old_weight * estimate_gaussian_mean + new_weight * estimate_gaussian_mean_grad

        # Var(sample) = E(sample**2) - E(sample)**2
        estimate_gaussian_var_grad = estimate_gaussian_square_grad - \
                                     torch.matmul(estimate_gaussian_mean_grad.unsqueeze(2), \
                                                  estimate_gaussian_mean_grad.unsqueeze(1))
        mean_diff = estimate_gaussian_mean - estimate_gaussian_mean_grad
        mean_diff_old = new_weight * mean_diff
        mean_diff_new = old_weight * mean_diff

        new_estimate_gaussian_var = old_weight * (estimate_gaussian_var + torch.matmul(mean_diff_old.unsqueeze(2), 
                                                                                    mean_diff_old.unsqueeze(1))) + \
                                    new_weight * (estimate_gaussian_var_grad + torch.matmul(mean_diff_new.unsqueeze(2), 
                                                                                    mean_diff_new.unsqueeze(1)))

        do_log_interpolate("\n====================== hook =======================\n")
        do_log_interpolate("ESTIMATE_GAUSSIAN: \n" + str(estimate_gaussian) + "\ngrad: \n" + str(estimate_gaussian.grad) + "\n weight: " + str(weight) + " weight.grad: " + str(weight.grad))
        estimate_gaussian[:, :D, :D] = new_estimate_gaussian_var
        estimate_gaussian[:, :D, -1] = new_estimate_gaussian_mean

        weight[:] = (weight_decay * weight + weight.grad)
        
        do_log_interpolate("\nafter ESTIMATE_GAUSSIAN: \n" + str(estimate_gaussian) + "\n weight: " + str(weight))

        estimate_gaussian.grad = None
        weight.grad = None
    
    return



# test
def test_interpolate(test_data_path, interpolate_method="vanilla"):
    '''
    ### D dimension of the space, E embedding length, B batchsize, S whole sample size
    end: Tensor[D+1, D]
    end_value: Tensor[D+1, E]
    sample: Tensor[B, D]
    estimate_end_value: Tensor[S, D+1, E]
    pred: Tensor[B, E]
    '''
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    test_data = torch.load(DATA_DIR, map_location=torch.device("cuda:0"))
    sample = test_data["sample"]
    label = test_data["label"]
    print("start testing: sample shape = ", sample.shape, ", label shape = ", label.shape)
    end=torch.tensor([[[-3],[3]]], dtype=torch.float32, device=torch.device("cuda:0"))
    end_value = torch.zeros(1,2,1, dtype=torch.float32, device=torch.device("cuda:0"), requires_grad=True)
    optimizer = SGD([end_value], lr=0.1)
    global ESTIMATE_GAUSSIAN, WEIGHT
    estimate_end_value = torch.zeros(sample.shape[0], 2, 1, device=torch.device("cuda:0"))
    if interpolate_method == "adaptive":
        estimate_gaussian_log = ESTIMATE_GAUSSIAN.detach().expand(sample.shape[0], -1, -1).contiguous()
    batchsize = 1

    for i in range(sample.shape[0]):
        if i % 100 == 0:
            print("\titeration ", i)
        gaussian_before_train_hook(ESTIMATE_GAUSSIAN, WEIGHT)
        optimizer.zero_grad()

        sampled_end = end[:].expand(batchsize, -1, -1)
        sampled_end_value = end_value[:].expand(batchsize, -1, -1)

        if interpolate_method == "vanilla":
            pred = vanilla_interpolate(sampled_end, sampled_end_value, sample[i:i+batchsize])
        elif interpolate_method == "adaptive":
            pred = adaptive_interpolation(sampled_end, sampled_end_value, sample[i:i+batchsize], 
                                            ESTIMATE_GAUSSIAN.expand(batchsize, -1, -1), 
                                            WEIGHT.expand(batchsize))
            estimate_gaussian_log[i] = ESTIMATE_GAUSSIAN

        loss = torch.sum((pred-label[i:i+batchsize])**2)
        loss.backward()
        optimizer.step()

        estimate_end_value[i] = end_value

    save_dict = {"est": estimate_end_value, "end": end[0]}
    if interpolate_method == "vanilla":
        torch.save(save_dict, os.path.join(SAVE_DIR, "pred.pt"))
    elif interpolate_method == "adaptive":
        save_dict["gaussian"] = estimate_gaussian_log
        torch.save(save_dict, os.path.join(SAVE_DIR, "pred_adaptive.pt"))


def do_log_interpolate(info):
    log_dir = "/media/user11209/D/git_clone_file/nerfstudio/external/flexnerf/interpolate/test/pred_data/log"
    log_path = os.path.join(log_dir, "log.txt")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with open(log_path, "a") as log_file:
        log_file.write(info)


# vanilla version of interpolation
def vanilla_interpolate(end, end_value, sample):
    '''
    ### D dimension of the space, E embedding length, B batchsize
    end: Tensor[B, D+1, D]
    end_value: Tensor[B, D+1, E]
    sample: Tensor[B, D]
    ret: Tensor[B, E]
    '''
    end_1 = torch.ones(end.shape[0], end.shape[1], 1, dtype=end.dtype, device=end.device)
    end_D_pls_1 = torch.concat([end, end_1], dim=2)                                      # [B, D+1, D+1]
    sample_1 = torch.ones(sample.shape[0], 1, dtype=sample.dtype, device=sample.device)
    sample_D_pls_1 = torch.concat([sample, sample_1], dim=1).unsqueeze(1)                # [B, 1, D+1]

    sample_interpolate = torch.linalg.solve(end_D_pls_1, sample_D_pls_1, left=False)     # [B, 1, D+1]
    pred = torch.matmul(sample_interpolate, end_value).squeeze(1)                        # [B, E]
    return pred


# adaptive version of interpolation, forward is the same as vanilla, backward requires gaussian estimation
class AdaptiveInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, end, end_value, sample, estimate_gaussian, weight):
        '''
        ### D dimension of the space, E embedding length, B batchsize
        end: Tensor[B, D+1, D]
        end_value: Tensor[B, D+1, E]
        sample: Tensor[B, D]

        estimate_gaussian: Tensor[B, D, D+1]
        weight: Tensor[B]
        ret: Tensor[B, E]
        '''
        ctx.end = end
        ctx.end_value = end_value
        ctx.sample = sample
        ctx.estimate_gaussian = estimate_gaussian
        ctx.weight = weight

        torch.set_grad_enabled(True)
        ctx.output = vanilla_interpolate(end, end_value, sample)
        torch.set_grad_enabled(False)
        detached_output = ctx.output.detach()
        return detached_output

    @staticmethod
    def backward(ctx, g):
        # do backward for vanilla_interpolation
        grad_original = torch.autograd.grad(ctx.output, ctx.end_value, g)[0] # [B, D+1, E]

        end_1 = torch.ones(ctx.end.shape[0], ctx.end.shape[1], 1, dtype=ctx.end.dtype, device=ctx.end.device)
        end_D_pls_1 = torch.concat([ctx.end, end_1], dim=2)               # X in comments
        sigma_mu_matrix = get_sigma_mu_matrix(ctx.estimate_gaussian)      # M in comments

        #@note $$ d(loss)/d(value) = (X'^(-1) * M * X^(-1)) * value $$, where * is matmul
        #! T     := (X'^(-1) * M * X^(-1))
        #! T^(-1) = X * M^(-1) * X'
        # grad_modified = T^(-1) * grad_original
        right_1 = torch.matmul(torch.transpose(end_D_pls_1, 1,2), grad_original)
        right_2 = torch.linalg.solve(sigma_mu_matrix, right_1, left=True)
        grad_modified = torch.matmul(end_D_pls_1, right_2)

        # store changes to estimate_gaussian in its gradient field.
        #! store E(sample) and E(sample**2), assemble them into expectation and variance at optimizer step.
        batch_weight = torch.linalg.norm(g, dim=-1)                                  # [B]
        estimate_gaussian_mean_grad = torch.unsqueeze(ctx.sample * batch_weight.view(-1,1), 2)  # [B, D, 1]
        estimate_gaussian_square_grad = torch.matmul(ctx.sample.unsqueeze(2), ctx.sample.unsqueeze(1)) * \
                                        batch_weight.view(-1,1,1) # [B, D, D]

        estimate_gaussian_grad = torch.cat([estimate_gaussian_square_grad, 
                                            estimate_gaussian_mean_grad], dim=2) # [B, D, D+1]
        
        # return gradients for end_value and estimate_gaussian.
        do_log_interpolate("\n====================== backward =======================\n")
        do_log_interpolate("gaussian_grad: \n" + str(estimate_gaussian_grad) + "\nbatch_weight: \n" + str(batch_weight))
        return (None, grad_modified, None, estimate_gaussian_grad, batch_weight)


adaptive_interpolation = AdaptiveInterpolation.apply



if __name__ == "__main__":
    test_interpolate(DATA_DIR, interpolate_method="adaptive")