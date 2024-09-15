import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from scipy.stats import multivariate_normal

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')  # 'k' for black



class Track:
    def __init__(self, id, state, covariance):
        self.id = id
        self.state = np.array(state)  # [x, y, vx, vy]
        self.covariance = np.array(covariance)
        self.history = [self.state[:2]]

class Observation:
    def __init__(self, position):
        self.position = np.array(position)

def predict_track(track, dt=1.0):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.eye(4) * 0.1  # Process noise
    track.state = F @ track.state
    track.covariance = F @ track.covariance @ F.T + Q
    track.history.append(track.state[:2])

def calculate_association_probabilities(tracks, observations):
    num_tracks = len(tracks)
    num_obs = len(observations)
    
    likelihood_matrix = np.zeros((num_obs, num_tracks))
    for i, obs in enumerate(observations):
        for j, track in enumerate(tracks):
            predicted_obs = track.state[:2]
            S = track.covariance[:2, :2]
            likelihood_matrix[i, j] = multivariate_normal.pdf(obs.position, predicted_obs, S)
    
    total_likelihood = likelihood_matrix.sum() + 1e-10
    likelihood_matrix /= total_likelihood
    
    beta = 0.1  # Probability of false alarm or new target
    association_probs = np.zeros((num_obs, num_tracks + 1))
    for i in range(num_obs):
        normalization = beta + likelihood_matrix[i, :].sum()
        association_probs[i, :-1] = likelihood_matrix[i, :] / normalization
        association_probs[i, -1] = beta / normalization
    
    return association_probs

def update_tracks_jpda(tracks, observations, association_probs):
    for j, track in enumerate(tracks):
        innovation = np.zeros(2)
        for i, obs in enumerate(observations):
            innovation += association_probs[i, j] * (obs.position - track.state[:2])
        
        S = track.covariance[:2, :2]
        K = track.covariance[:, :2] @ np.linalg.inv(S)
        
        track.state += K @ innovation
        track.covariance -= K @ S @ K.T

def jpda_mht(tracks, observations):
    for track in tracks:
        predict_track(track)
    
    association_probs = calculate_association_probabilities(tracks, observations)
    update_tracks_jpda(tracks, observations, association_probs)
    
    return association_probs

class Simulation:
    def __init__(self):
        self.tracks = [
            Track(1, [0, 0, 1, 1], np.eye(4) * 0.5),
            Track(2, [10, 10, -1, -1], np.eye(4) * 0.5)
        ]
        self.time = 0
    
    def generate_observations(self):
        observations = []
        for track in self.tracks:
            if np.random.rand() < 0.9:  # 90% detection rate
                true_pos = track.state[:2] + np.random.normal(0, 0.5, 2)
                observations.append(Observation(true_pos))
        
        # Add false alarms
        num_false = np.random.poisson(0.5)
        for _ in range(num_false):
            false_pos = np.random.uniform(0, 20, 2)
            observations.append(Observation(false_pos))
        
        return observations
    
    def step(self):
        self.time += 1
        observations = self.generate_observations()
        jpda_mht(self.tracks, observations)
        return observations

class VisualizationWidget(pg.GraphicsLayoutWidget):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation
        self.plot = self.addPlot(title="JPDA-MHT Simulation")
        self.plot.setXRange(0, 20)
        self.plot.setYRange(0, 20)
        
        self.track_plots = {}
        self.observation_plot = self.plot.plot([], [], pen=None, symbol='o', symbolBrush='r')
        
        for track in self.simulation.tracks:
            self.track_plots[track.id] = self.plot.plot([], [], pen=pg.mkPen(color=(np.random.randint(256), np.random.randint(256), np.random.randint(256)), width=2))
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)  # Update every 100 ms
    
    def update(self):
        observations = self.simulation.step()
        
        obs_x = [obs.position[0] for obs in observations]
        obs_y = [obs.position[1] for obs in observations]
        self.observation_plot.setData(obs_x, obs_y)
        
        for track in self.simulation.tracks:
            history = np.array(track.history)
            self.track_plots[track.id].setData(history[:, 0], history[:, 1])

if __name__ == '__main__':
    simulation = Simulation()
    app = pg.mkQApp("JPDA-MHT Simulation")
    widget = VisualizationWidget(simulation)
    widget.show()
    app.exec_()