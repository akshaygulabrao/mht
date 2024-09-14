import numpy as np
from mht_implementation import MHT
import logging

def generate_trajectory(start, end, steps):
    return np.array([np.linspace(start[i], end[i], steps) for i in range(2)]).T

def add_noise(trajectory, noise_level):
    return trajectory + np.random.normal(0, noise_level, trajectory.shape)

def run_mht_simulation(true_trajectories, noise_level, detection_prob, false_alarm_rate, steps):
    mht = MHT(max_hypotheses=100, state_dim=4, meas_dim=2, pruning_threshold=1e-5)
    
    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.eye(4) * 0.1
    R = np.eye(2) * noise_level**2
    
    for step in range(steps):
        logging.info(f"Processing step {step + 1}/{steps}")
        
        # Generate noisy measurements
        measurements = []
        for traj_idx, traj in enumerate(true_trajectories):
            if np.random.random() < detection_prob:
                meas = traj[step, :2] + np.random.normal(0, noise_level, 2)
                measurements.append((traj_idx, meas))
        
        # Add false alarms
        num_false_alarms = np.random.poisson(false_alarm_rate)
        for _ in range(num_false_alarms):
            false_alarm = np.random.uniform(0, 100, 2)
            measurements.append((-1, false_alarm))  # -1 indicates false alarm
        
        logging.info(f"Generated {len(measurements)} measurements ({num_false_alarms} false alarms)")
        logging.info(f"Measurements: {measurements}")
        
        # MHT update
        mht.predict(F, Q)
        mht.update([m[1] for m in measurements], R, detection_prob, false_alarm_rate)
        
        # Log all hypotheses
        logging.info(f"Step {step + 1}: All hypotheses:")
        for i, hypothesis in enumerate(mht.hypotheses):
            logging.info(f"Hypothesis {i + 1}: weight = {hypothesis['weight']:.4f}")
            for j, track in enumerate(hypothesis['tracks']):
                position = track['state'][:2]
                velocity = track['state'][2:]
                measurement_idx = track.get('measurement_idx', None)
                if measurement_idx is not None:
                    true_traj_idx, meas = measurements[measurement_idx]
                    logging.info(f"  Track {j + 1}: position = ({position[0]:.2f}, {position[1]:.2f}), "
                                 f"velocity = ({velocity[0]:.2f}, {velocity[1]:.2f}), "
                                 f"associated with measurement {measurement_idx + 1} "
                                 f"(true trajectory: {true_traj_idx}, measurement: {meas})")
                else:
                    logging.info(f"  Track {j + 1}: position = ({position[0]:.2f}, {position[1]:.2f}), "
                                 f"velocity = ({velocity[0]:.2f}, {velocity[1]:.2f}), "
                                 f"no associated measurement")

        # Log best hypothesis
        best_hypothesis = mht.get_best_hypothesis()
        logging.info(f"Step {step + 1}: Most likely hypothesis (weight: {best_hypothesis['weight']:.4f}):")
        for i, track in enumerate(best_hypothesis['tracks']):
            position = track['state'][:2]
            velocity = track['state'][2:]
            measurement_idx = track.get('measurement_idx', None)
            if measurement_idx is not None:
                true_traj_idx, meas = measurements[measurement_idx]
                logging.info(f"Track {i + 1}: position = ({position[0]:.2f}, {position[1]:.2f}), "
                             f"velocity = ({velocity[0]:.2f}, {velocity[1]:.2f}), "
                             f"associated with measurement {measurement_idx + 1} "
                             f"(true trajectory: {true_traj_idx}, measurement: {meas})")
            else:
                logging.info(f"Track {i + 1}: position = ({position[0]:.2f}, {position[1]:.2f}), "
                             f"velocity = ({velocity[0]:.2f}, {velocity[1]:.2f}), "
                             f"no associated measurement")

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='mht_simulation_detailed.log',
        filemode='w', level=logging.DEBUG)
    
    np.random.seed(42)
    steps = 50
    noise_level = 1.0
    detection_prob = 0.9
    false_alarm_rate = 0.1

    true_trajectories = [
        generate_trajectory((10, 10), (80, 80), steps),
        generate_trajectory((90, 10), (20, 80), steps),
        generate_trajectory((50, 90), (50, 10), steps)
    ]

    run_mht_simulation(true_trajectories, noise_level, detection_prob, false_alarm_rate, steps)