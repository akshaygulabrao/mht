import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from scipy.stats import multivariate_normal

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')  # 'k' for black


class Simulation:
    def __init__(self):
        # Initialize true states of objects
        # ...
        pass

    def step(self):
        # Update true states
        # Generate noisy observations
        return observations

    def get_true_states(self):
        # Return the ground truth states
        # ...
        return true_states

class Tracker:
    def __init__(self):
        # Initialize tracker state
        # ...
        pass

    def update(self, observations):
        # Process new observations
        # Update tracks
        return estimated_states

# Main loop
simulation = Simulation()
tracker = Tracker()

for t in range(1000):
    observations = simulation.step()
    estimated_states = tracker.update(observations)
    
    # Evaluation (outside of both simulation and tracker)
    true_states = simulation.get_true_states()
    evaluate_performance(true_states, estimated_states)

class VisualizationWidget(pg.GraphicsLayoutWidget):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation
        self.plot = self.addPlot(title="JPDA-MHT Simulation")
        self.plot.setXRange(0, 20)
        self.plot.setYRange(0, 20)
        
        self.track_plots = {}
        self.observation_plot = self.plot.plot([], [], pen=None, symbol='o', symbolBrush='r')
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)  # Update every 100 ms
    
    def update(self):
        observations = self.simulation.step()
        
        obs_x = [obs.position[0] for obs in observations]
        obs_y = [obs.position[1] for obs in observations]
        self.observation_plot.setData(obs_x, obs_y)
        
        # Update existing tracks and add new ones
        for track in self.simulation.tracks:
            if track.id not in self.track_plots:
                self.track_plots[track.id] = self.plot.plot([], [], pen=pg.mkPen(color=(np.random.randint(256), np.random.randint(256), np.random.randint(256)), width=2))
            history = np.array(track.history)
            self.track_plots[track.id].setData(history[:, 0], history[:, 1])
        
        # Remove terminated tracks
        current_track_ids = set(track.id for track in self.simulation.tracks)
        for track_id in list(self.track_plots.keys()):
            if track_id not in current_track_ids:
                self.plot.removeItem(self.track_plots[track_id])
                del self.track_plots[track_id]

if __name__ == '__main__':
    simulation = Simulation()
    app = pg.mkQApp("JPDA-MHT Simulation")
    widget = VisualizationWidget(simulation)
    widget.show()
    app.exec_()