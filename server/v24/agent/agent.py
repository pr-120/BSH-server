import numpy as np
import tensorflow as tf
import pandas as pd
import os
from environment.anomaly_detection.constructor import get_preprocessor
from environment.state_handling import get_num_configs
from environment.settings import (
    ALL_CSV_HEADERS,
    TRAINING_CSV_FOLDER_PATH,
    LEARN_RATE_V24,
    CLIP_EPSILON_V24,
    GAMMA_V24,
    LAMBDA_V24,
    VALUE_COEF_V24,
    ENTROPY_COEF_INITIAL_V24,
    ENTROPY_COEF_DECAY_V24,
    MIN_ENTROPY_COEF_V24,
    EPOCHS_V24,
    BATCH_SIZE_V24,
)

class AgentPPONormalAD(tf.keras.Model):
    def __init__(self, representation=None, current_episode=0):
        super(AgentPPONormalAD, self).__init__()
        self.current_episode = current_episode

        # Define headers once for consistency
        self.headers = ALL_CSV_HEADERS.split(",")
        self.num_headers = len(self.headers)

        if representation:
            # Load from representation
            self.num_input = representation["num_input"]
            self.num_hidden1 = representation["num_hidden1"]
            self.num_hidden2 = representation["num_hidden2"]
            self.num_output = representation["num_output"]
            self.actions = list(range(self.num_output))
            self.learn_rate = representation["learn_rate"]
            self.clip_epsilon = representation["clip_epsilon"]
            self.gamma = representation["gamma"]
            self.lambda_ = representation["lambda_"]
            self.value_coef = representation["value_coef"]
            self.entropy_coef_initial = representation["entropy_coef_initial"]
            self.entropy_coef_decay = representation["entropy_coef_decay"]
            self.min_entropy_coef = representation["min_entropy_coef"]
            self.epochs = representation["epochs"]
            self.batch_size = representation["batch_size"]
            self.fp_features = representation["fp_features"]
            self.min = np.array(representation["min"])
            self.max = np.array(representation["max"])
            # Build model and set weights
            self._build_model()
            dummy_input = tf.zeros((1, self.num_headers), dtype=tf.float32)
            self.call(dummy_input)
            self.set_weights_from_dict(representation)
        else:
            # Initialize with default parameters
            num_configs = get_num_configs()
            self.actions = list(range(num_configs))
            # Load raw data and compute min/max for all features
            raw_df = pd.read_csv(os.path.join(TRAINING_CSV_FOLDER_PATH, "normal-behavior.csv"))
            self.min = raw_df.min().values
            self.max = raw_df.max().values
            # Get preprocessed feature list
            preprocessor = get_preprocessor()
            ready_dataset = preprocessor.preprocess_dataset(raw_df)
            self.fp_features = ready_dataset.columns.tolist()
            self.num_input = len(self.fp_features)
            self.num_hidden1 = round(self.num_input * 2)
            self.num_hidden2 = round(self.num_input * 0.5)
            self.num_output = num_configs
            self.learn_rate = LEARN_RATE_V24
            self.clip_epsilon = CLIP_EPSILON_V24
            self.gamma = GAMMA_V24
            self.lambda_ = LAMBDA_V24
            self.value_coef = VALUE_COEF_V24
            self.entropy_coef_initial = ENTROPY_COEF_INITIAL_V24
            self.entropy_coef_decay = ENTROPY_COEF_DECAY_V24
            self.min_entropy_coef = MIN_ENTROPY_COEF_V24
            self.epochs = EPOCHS_V24
            self.batch_size = BATCH_SIZE_V24
            self._build_model()
            dummy_input = tf.zeros((1, self.num_headers), dtype=tf.float32)
            self.call(dummy_input)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)

    def _build_model(self):
        """Define the neural network architecture."""
        self.hidden1 = tf.keras.layers.Dense(
            self.num_hidden1, activation=tf.nn.leaky_relu,
            kernel_initializer='he_normal'
        )
        self.hidden2 = tf.keras.layers.Dense(
            self.num_hidden2, activation=tf.nn.leaky_relu,
            kernel_initializer='he_normal'
        )
        self.policy = tf.keras.layers.Dense(
            self.num_output, activation=None,
            kernel_initializer='he_normal'
        )
        self.value = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer='he_normal'
        )

    def set_weights_from_dict(self, representation):
        """Set model weights from a dictionary."""
        self.hidden1.kernel.assign(np.array(representation["weights_input_hidden1"]))
        self.hidden1.bias.assign(np.array(representation["bias_hidden1"]))
        self.hidden2.kernel.assign(np.array(representation["weights_hidden1_hidden2"]))
        self.hidden2.bias.assign(np.array(representation["bias_hidden2"]))
        self.policy.kernel.assign(np.array(representation["weights_hidden2_policy"]))
        self.policy.bias.assign(np.array(representation["bias_policy"]))
        self.value.kernel.assign(np.array(representation["weights_hidden2_value"]))
        self.value.bias.assign(np.array(representation["bias_value"]))

    def get_weights_dict(self):
        """Return weights as a dictionary for saving."""
        return {
            "num_input": self.num_input,
            "num_hidden1": self.num_hidden1,
            "num_hidden2": self.num_hidden2,
            "num_output": self.num_output,
            "learn_rate": self.learn_rate,
            "clip_epsilon": self.clip_epsilon,
            "gamma": self.gamma,
            "lambda_": self.lambda_,
            "value_coef": self.value_coef,
            "entropy_coef_initial": self.entropy_coef_initial,
            "entropy_coef_decay": self.entropy_coef_decay,
            "min_entropy_coef": self.min_entropy_coef,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weights_input_hidden1": self.hidden1.kernel.numpy().tolist(),
            "bias_hidden1": self.hidden1.bias.numpy().tolist(),
            "weights_hidden1_hidden2": self.hidden2.kernel.numpy().tolist(),
            "bias_hidden2": self.hidden2.bias.numpy().tolist(),
            "weights_hidden2_policy": self.policy.kernel.numpy().tolist(),
            "bias_policy": self.policy.bias.numpy().tolist(),
            "weights_hidden2_value": self.value.kernel.numpy().tolist(),
            "bias_value": self.value.bias.numpy().tolist(),
            "fp_features": self.fp_features,
            "min": self.min.tolist(),
            "max": self.max.tolist()
        }

    @property
    def entropy_coef(self):
        """Calculate the decayed entropy coefficient with a minimum value."""
        return max(self.min_entropy_coef, self.entropy_coef_initial * (self.entropy_coef_decay ** self.current_episode))

    def _preprocess_fp(self, fp):
        """Select relevant features from the fingerprint."""
        indexes = [self.headers.index(header) for header in self.fp_features if header in self.headers]
        return tf.gather(fp, indexes, axis=-1)

    def standardize_fp(self, fp):
        """Standardize the fingerprint using min-max scaling on full input."""
        if tf.rank(fp) == 1:
            fp = tf.expand_dims(fp, 0)
        return (fp - self.min) / (self.max - self.min + 1e-6)

    def call(self, inputs, training=False):
        """Forward pass through the network."""
        x = self.standardize_fp(inputs)  # Standardize full input first
        x = self._preprocess_fp(x)       # Then select features
        hidden1 = self.hidden1(x)
        hidden2 = self.hidden2(hidden1)
        policy_logits = self.policy(hidden2)
        value = self.value(hidden2)
        return policy_logits, value

    def act(self, state):
        """Select an action stochastically based on the policy."""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if state.shape[0] != self.num_headers:
            state = tf.pad(state, [[0, self.num_headers - state.shape[0]]], mode='CONSTANT') if state.shape[0] < self.num_headers else state[:self.num_headers]
        state = tf.expand_dims(state, 0)  # Add batch dimension: (1, num_headers)

        policy_logits, value = self.call(state)
        probs = tf.nn.softmax(policy_logits)[0]
        action = tf.random.categorical(tf.expand_dims(probs, 0), 1)[0, 0]
        action = tf.clip_by_value(action, 0, self.num_output - 1)
        log_prob = tf.math.log(probs[action] + 1e-10)
        return int(action), float(log_prob), float(value[0])

    def evaluate_action(self, state):
        """Deterministically select the most probable action for evaluation."""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if state.shape[0] != self.num_headers:
            state = tf.pad(state, [[0, self.num_headers - state.shape[0]]], mode='CONSTANT') if state.shape[0] < self.num_headers else state[:self.num_headers]
        state = tf.expand_dims(state, 0)
        policy_logits, _ = self.call(state)
        probs = tf.nn.softmax(policy_logits)[0]  # Use softmax probabilities
        return int(tf.argmax(probs))

    def update(self, states, actions, log_probs_old, old_values, advantages, returns):
        """Update the policy and value networks using PPO with TensorFlow."""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        log_probs_old = tf.convert_to_tensor(log_probs_old, dtype=tf.float32)
        old_values = tf.convert_to_tensor(old_values, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        for _ in range(self.epochs):
            indices = tf.random.shuffle(tf.range(tf.shape(states)[0]))
            states = tf.gather(states, indices)
            actions = tf.gather(actions, indices)
            log_probs_old = tf.gather(log_probs_old, indices)
            old_values = tf.gather(old_values, indices)
            advantages = tf.gather(advantages, indices)
            returns = tf.gather(returns, indices)

            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_log_probs_old = log_probs_old[start:end]
                batch_old_values = old_values[start:end]
                batch_advantages = advantages[start:end]
                batch_returns = returns[start:end]

                with tf.GradientTape() as tape:
                    policy_logits, value = self.call(batch_states)
                    # Stable log probability calculation
                    log_probs_new = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=batch_actions, logits=policy_logits
                    )
                    ratio = tf.exp(log_probs_new - batch_log_probs_old)
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -tf.minimum(surr1, surr2)
                    probs = tf.nn.softmax(policy_logits)
                    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
                    # Value loss with clipping
                    value_pred_clipped = batch_old_values + tf.clip_by_value(
                        value - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                    )
                    value_losses = tf.square(value - batch_returns)
                    value_losses_clipped = tf.square(value_pred_clipped - batch_returns)
                    value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_losses, value_losses_clipped))
                    total_loss = tf.reduce_mean(policy_loss - self.entropy_coef * entropy) + self.value_coef * value_loss

                gradients = tape.gradient(total_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))