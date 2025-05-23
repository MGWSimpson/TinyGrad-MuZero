import pickle
from tinygrad import Tensor, nn
import ray
import time
from core.data_worker import DataWorker
from core.shared_storage import SharedStorage
from core.replay_buffer import ReplayBuffer
from core.utils import soft_update
from core.test import test
import logging


train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_test')



def _log(config, step_count, log_data, model, replay_buffer, lr,):
    loss_data = log_data
    loss, policy_loss, reward_loss, value_loss = loss_data



    _msg = '#{:<10} Loss: {:<8.3f} [Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
           'Reward Loss: {:<8.3f} ] Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Lr: {:<8.3f}'
    _msg = _msg.format(step_count, loss, policy_loss, value_loss, reward_loss,
                       0, 0, lr)
    train_logger.info(_msg)



def _train(config, shared_storage, replay_buffer):

    model = config.get_uniform_network()
    model.train()

    target_model = config.get_uniform_network()
    target_model.eval()



    optim = nn.optim.Adam(params=nn.state.get_parameters([model._dynamics_reward, model._prediction_actor, model._prediction_value]), lr=0.001)



    while ray.get(replay_buffer.size.remote()) == 0:
        pass


    for step_count in range(config.training_steps):

        shared_storage.incr_counter.remote()


        if step_count % config.checkpoint_interval == 0:  ## after a certain number of training steps, save the model so the workers can use it
            shared_storage.set_weights.remote(model.get_weights())

        log_data = update_weights(model, target_model, optim, replay_buffer, config)

        # softly update target model
        if config.use_target_model:
            soft_update(target_model, model, tau=1e-2)
            target_model.eval()

        _log(config, step_count, log_data, model, replay_buffer, config.lr_init)
        

        if step_count % 50 == 0:
            replay_buffer.remove_to_fit.remote()


    shared_storage.set_weights.remote(model.get_weights())


def update_weights(model, target_model, optim, replay_buffer, config):

    batch = ray.get(replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps,
                                                      model=None,
                                                      config=config))

    obs_batch, action_batch, target_reward, target_value, target_policy, indices = batch

    obs_batch = Tensor(obs_batch)
    action_batch = Tensor(action_batch)
    target_reward = Tensor(target_reward)
    target_value = Tensor(target_value)
    target_policy = Tensor(target_policy)

    
    


    obs_batch = obs_batch.to(config.device)
    action_batch = action_batch.to(config.device).unsqueeze(-1)
    target_reward = target_reward.to(config.device)
    target_value = target_value.to(config.device)
    target_policy = target_policy.to(config.device)

  
    

    
    # transform targets to categorical rep
    transformed_target_reward = config.scalar_transform(target_reward)
    target_reward_phi = config.reward_phi(transformed_target_reward)
    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)





    value, reward, policy_logits, hidden_state = model.initial_inference(obs_batch)
    
   
    
    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(Tensor.log_softmax(policy_logits, axis=1) * target_policy[:, 0]).sum(1)
    reward_loss = Tensor.zeros(config.batch_size)

    optim = nn.optim.Adam(params=[value_loss, policy_loss, reward_loss], lr=0.001)


    for step_i in range(config.num_unroll_steps):
        value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])
        policy_loss += -(Tensor.log_softmax(policy_logits, axis=1) * target_policy[:, step_i + 1]).sum(1)
        value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
        reward_loss += config.scalar_reward_loss(reward, target_reward_phi[:, step_i])
        # register hook...


    optim.zero_grad()
    # compute loss
    gradient_scale = 1 / config.num_unroll_steps
    # optimize
    loss = (policy_loss + config.value_loss_coeff * value_loss + reward_loss)
    
    
    loss = loss.mean()


    loss.backward()
    optim.step()

   

    loss_data = (loss.item(), policy_loss.mean().item(), reward_loss.mean().item(),
                 value_loss.mean().item())
    

    return loss_data

@ray.remote
def _test(config, shared_storage): 
    test_model = config.get_uniform_network()
    best_test_score = float('-inf')
   
    while ray.get(shared_storage.get_counter.remote()) < config.training_steps:
        test_model.set_weights(ray.get(shared_storage.get_weights.remote()))
        test_model.eval()
        test_score = test(config, test_model, config.test_episodes, 'cpu', False)
        if test_score >= best_test_score:
            best_test_score = test_score
            test_model_state_dict = nn.state.get_state_dict(test_model)
            #nn.state.safe_save(test_model_state_dict, config.model_path)
            
        time.sleep(30)   



def train(config,):

    storage = SharedStorage.remote(config.get_uniform_network().get_weights())
    replay_buffer = ReplayBuffer.remote(batch_size=config.batch_size, capacity=config.window_size)
    workers = [DataWorker.remote(rank, config, storage, replay_buffer)
               for rank in range(config.num_actors)]
    worker_waitables = [worker.run.remote() for worker in workers]
    worker_waitables += [_test.remote(config, storage)]
    _train(config, storage, replay_buffer)
    ray.wait(ray_waitables=worker_waitables, num_returns=len(worker_waitables))
   
