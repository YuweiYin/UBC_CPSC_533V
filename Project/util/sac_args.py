class SACArgs:

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def __contains__(self, key):
        return hasattr(self, key)

    env_name = 'HalfCheetah-v4'  # @param ["HalfCheetah-v4",'CartPole-v0', 'InvertedPendulum-v2', 'HalfCheetah-v2']
    exp_name = "sac"  # @param

    # PDF will tell you how to set ep_len and discount for each environment
    ep_len = 150  # @param {type: "integer"}

    # @markdown batches and steps
    train_batch_size = 1000
    batch_size = 1000  # @param {type: "integer"}
    eval_batch_size = 400  # @param {type: "integer"}

    n_iter = 20000  # @param {type: "integer"}
    num_agent_train_steps_per_iter = 1  # @param {type: "integer"}
    num_actor_updates_per_agent_update = 1  # @param {type: "integer"}
    num_critic_updates_per_agent_update = 1  # @param {type: "integer"}
    actor_update_frequency = 1  # @param {type: "integer"}
    critic_target_update_frequency = 1  # @param {type: "integer"}
    # @markdown Actor-Critic parameters
    discount = 0.99  # @param {type: "number"}
    learning_rate = 3e-4  # @param {type: "number"}
    n_layers = 2  # @param {type: "integer"}
    size = 64  # @param {type: "integer"}

    # @markdown system
    save_params = False  # @param {type: "boolean"}
    no_gpu = False  # @param {type: "boolean"}
    which_gpu = 0  # @param {type: "integer"}
    seed = 1000  # @param {type: "integer"}

    # @markdown logging
    # default is to not log video so that logs are small enough to be uploaded to gradscope
    video_log_freq = -1  # @param {type: "integer"}
    scalar_log_freq = 10  # @param {type: "integer"}
    init_temperature = 0.1
