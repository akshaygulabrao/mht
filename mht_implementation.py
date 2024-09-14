import numpy as np
from scipy.stats import multivariate_normal
from itertools import product

class MHT:
    def __init__(self, max_hypotheses=100, state_dim=4, meas_dim=2):
        self.max_hypotheses = max_hypotheses
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.hypotheses = [{'tracks': [], 'weight': 1.0}]
        
    def predict(self, F, Q):
        for hypothesis in self.hypotheses:
            for track in hypothesis['tracks']:
                track['state'] = F @ track['state']
                track['covariance'] = F @ track['covariance'] @ F.T + Q
    
    def update(self, measurements, R, detection_prob=0.9, false_alarm_rate=1e-6):
        new_hypotheses = []
        H = np.zeros((self.meas_dim, self.state_dim))
        H[:self.meas_dim, :self.meas_dim] = np.eye(self.meas_dim)  # Observation matrix
        
        for hypothesis in self.hypotheses:
            # Generate all possible measurement-to-track assignments
            assignments = list(product(range(len(measurements) + 1), repeat=len(hypothesis['tracks'])))
            
            for assignment in assignments:
                new_hypothesis = {'tracks': [], 'weight': hypothesis['weight']}
                
                for i, track in enumerate(hypothesis['tracks']):
                    if assignment[i] < len(measurements):
                        # Update track with measurement
                        z = measurements[assignment[i]]
                        y = z - H @ track['state']
                        S = H @ track['covariance'] @ H.T + R
                        K = track['covariance'] @ H.T @ np.linalg.inv(S)
                        new_state = track['state'] + K @ y
                        new_covariance = (np.eye(self.state_dim) - K @ H) @ track['covariance']
                        
                        new_hypothesis['tracks'].append({
                            'state': new_state,
                            'covariance': new_covariance
                        })
                        
                        # Update hypothesis weight
                        new_hypothesis['weight'] *= detection_prob * multivariate_normal.pdf(y, cov=S)
                    else:
                        # Track not associated with any measurement
                        new_hypothesis['tracks'].append(track.copy())
                        new_hypothesis['weight'] *= (1 - detection_prob)
                
                # Handle unassigned measurements (potential new tracks)
                for j in range(len(measurements)):
                    if j not in assignment:
                        initial_state = np.zeros(self.state_dim)
                        initial_state[:self.meas_dim] = measurements[j]
                        new_hypothesis['tracks'].append({
                            'state': initial_state,
                            'covariance': np.eye(self.state_dim) * 100  # Initial uncertainty
                        })
                        new_hypothesis['weight'] *= false_alarm_rate
                
                new_hypotheses.append(new_hypothesis)
        
        # Normalize weights and prune hypotheses
        total_weight = sum(h['weight'] for h in new_hypotheses)
        for h in new_hypotheses:
            h['weight'] /= total_weight
        
        self.hypotheses = sorted(new_hypotheses, key=lambda h: h['weight'], reverse=True)[:self.max_hypotheses]

    def get_best_hypothesis(self):
        return max(self.hypotheses, key=lambda h: h['weight'])