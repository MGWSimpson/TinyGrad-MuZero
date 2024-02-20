from tinygrad import Tensor, nn
from config.classic_control import MuZeroNet
from config.classic_control import DiscreteSupport
from config.classic_control import ClassicControlConfig
import argparse


parser = argparse.ArgumentParser(description='MuZero Pytorch Implementation')
parser.add_argument('--env', required=True, help='Name of the environment')
parser.add_argument('--num_actors', type=int, default=1,
                        help='Number of actors running concurrently (default: %(default)s)')
parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')


args = parser.parse_args()


config = ClassicControlConfig(args=args)

value_support=DiscreteSupport(-20, 20)
reward_support=DiscreteSupport(-5, 5)

model = MuZeroNet(input_size=4, action_space_n=2, value_support_size=value_support.size, reward_support_size=reward_support.size)



optim = nn.optim.Adam(params=nn.state.get_parameters([model._dynamics_reward, model._prediction_actor, model._prediction_value]), lr=0.001)






obs_batch = Tensor.rand(4).unsqueeze(0)
action_batch = Tensor([[[1]]])
target_reward = Tensor([[1], [1]])
target_value = Tensor([[1], [1]])
target_policy = Tensor([[0.5, 0.5], [0.5, 0.5]])



# transform targets to categorical rep
transformed_target_reward = config.scalar_transform(target_reward)
target_reward_phi = config.reward_phi(transformed_target_reward)
transformed_target_value = config.scalar_transform(target_value)
target_value_phi = config.value_phi(transformed_target_value)




value, _, policy_logits, hidden_state = model.initial_inference(obs_batch)
    
value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
policy_loss = -(Tensor.log_softmax(policy_logits, axis=1) * target_policy[:, 0]).sum(1)
reward_loss = Tensor.zeros(config.batch_size, device=config.device)




for step_i in range(1 ):
    
    value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])
    policy_loss += -(Tensor.log_softmax(policy_logits, axis=1) * target_policy[:, step_i + 1]).sum(1)
    value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
    reward_loss += config.scalar_reward_loss(reward, target_reward_phi[:, step_i])
        
    
   


    
optim.zero_grad()

loss = (value_loss + reward_loss + policy_loss)

loss = loss.mean()
    
loss.backward()

 
optim.step()
print(i, loss.item())







def scale_gradient(tensor, scale):
    
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)




