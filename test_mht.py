import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mht_implementation import MHT  # Assuming the updated code is in mht_implementation.py

def generate_trajectory(start, end, steps):
    return np.array([np.linspace(start[i], end[i], steps) for i in range(2)]).T

def add_noise(trajectory, noise_level):
    return trajectory + np.random.normal(0, noise_level, trajectory.shape)

class MHTSimulation:
    def __init__(self, true_trajectories, noise_level, detection_prob, false_alarm_rate, steps):
        self.true_trajectories = true_trajectories
        self.noise_level = noise_level
        self.detection_prob = detection_prob
        self.false_alarm_rate = false_alarm_rate
        self.steps = steps
        self.current_step = 0
        
        self.mht = MHT(max_hypotheses=100, state_dim=4, meas_dim=2)
        self.measurements = []
        self.mht_results = []
        
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * noise_level**2

    def step(self):
        if self.current_step >= self.steps:
            return False

        # Generate noisy measurements
        step_measurements = []
        for traj in self.true_trajectories:
            if np.random.random() < self.detection_prob:
                meas = traj[self.current_step, :2] + np.random.normal(0, self.noise_level, 2)
                step_measurements.append(meas)
        
        # Add false alarms
        num_false_alarms = np.random.poisson(self.false_alarm_rate)
        for _ in range(num_false_alarms):
            false_alarm = np.random.uniform(0, 100, 2)
            step_measurements.append(false_alarm)
        
        # MHT update
        self.mht.predict(self.F, self.Q)
        self.mht.update(step_measurements, self.R, self.detection_prob, self.false_alarm_rate)
        
        # Store results
        best_hypothesis = self.mht.get_best_hypothesis()
        self.measurements.append(step_measurements)
        self.mht_results.append([track['state'][:2] for track in best_hypothesis['tracks']])
        
        self.current_step += 1
        return True

def animate_mht(true_trajectories, noise_level, detection_prob, false_alarm_rate, steps):
    simulation = MHTSimulation(true_trajectories, noise_level, detection_prob, false_alarm_rate, steps)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title('Multiple Hypothesis Tracker Animation')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    true_lines = [ax.plot([], [], 'k-', label='True')[0] for _ in true_trajectories]
    meas_scatter = ax.scatter([], [], c='r', s=10, label='Measurements')
    mht_lines = [ax.plot([], [], 'g--', label='MHT')[0] for _ in range(len(true_trajectories))]
    
    ax.legend()
    
    def init():
        for line in true_lines + mht_lines:
            line.set_data([], [])
        meas_scatter.set_offsets(np.empty((0, 2)))
        return true_lines + [meas_scatter] + mht_lines
    
    def update(frame):
        simulation.step()
        
        # Update true trajectories
        for i, line in enumerate(true_lines):
            line.set_data(true_trajectories[i][:frame+1, 0], true_trajectories[i][:frame+1, 1])
        
        # Update measurements
        all_measurements = np.concatenate(simulation.measurements)
        meas_scatter.set_offsets(all_measurements)
        
        # Update MHT results
        for i, line in enumerate(mht_lines):
            track_results = [result[i] for result in simulation.mht_results if i < len(result)]
            if track_results:
                line.set_data([state[0] for state in track_results], [state[1] for state in track_results])
        
        return true_lines + [meas_scatter] + mht_lines
    
    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=100)
    plt.show()

# Run simulation
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

animate_mht(true_trajectories, noise_level, detection_prob, false_alarm_rate, steps)