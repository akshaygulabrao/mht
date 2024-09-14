import numpy as np
from scipy.stats import multivariate_normal
from itertools import product
import logging

class MHT:
    def __init__(self, max_hypotheses=100, state_dim=4, meas_dim=2, pruning_threshold=1e-5):
        self.max_hypotheses = max_hypotheses
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.pruning_threshold = pruning_threshold
        self.hypotheses = [{'tracks': [], 'weight': 1.0}]
        logging.info(f"Initialized MHT with max_hypotheses={max_hypotheses}, state_dim={state_dim}, meas_dim={meas_dim}")
        
    def predict(self, F, Q):
        logging.debug(f"Predicting with {len(self.hypotheses)} hypotheses")
        for hypothesis in self.hypotheses:
            for track in hypothesis['tracks']:
                track['state'] = F @ track['state']
                track['covariance'] = F @ track['covariance'] @ F.T + Q
        logging.debug("Prediction completed")
    
    def update(self, measurements, R, detection_prob=0.9, false_alarm_rate=1e-6):
        logging.debug(f"Updating with {len(measurements)} measurements")
        logging.debug(f"Measurements: {measurements}")
        new_hypotheses = []
        H = np.zeros((self.meas_dim, self.state_dim))
        H[:self.meas_dim, :self.meas_dim] = np.eye(self.meas_dim)  # Observation matrix
        
        for hypothesis_idx, hypothesis in enumerate(self.hypotheses):
            logging.debug(f"Processing hypothesis {hypothesis_idx + 1}/{len(self.hypotheses)}")
            # Generate all possible measurement-to-track assignments
            assignments = list(product(range(len(measurements) + 1), repeat=len(hypothesis['tracks'])))
            
            for assignment_idx, assignment in enumerate(assignments):
                new_hypothesis = {'tracks': [], 'weight': hypothesis['weight'], 'assignment': assignment}
                
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
                            'covariance': new_covariance,
                            'measurement_idx': assignment[i]
                        })
                        
                        # Update hypothesis weight
                        new_hypothesis['weight'] *= detection_prob * multivariate_normal.pdf(y, cov=S)
                    else:
                        # Track not associated with any measurement
                        new_hypothesis['tracks'].append({**track.copy(), 'measurement_idx': None})
                        new_hypothesis['weight'] *= (1 - detection_prob)
                
                # Handle unassigned measurements (potential new tracks)
                for j in range(len(measurements)):
                    if j not in assignment:
                        initial_state = np.zeros(self.state_dim)
                        initial_state[:self.meas_dim] = measurements[j]
                        new_hypothesis['tracks'].append({
                            'state': initial_state,
                            'covariance': np.eye(self.state_dim) * 100,  # Initial uncertainty
                            'measurement_idx': j
                        })
                        new_hypothesis['weight'] *= false_alarm_rate
                
                new_hypotheses.append(new_hypothesis)
                logging.debug(f"Created new hypothesis {assignment_idx + 1}/{len(assignments)} with weight {new_hypothesis['weight']}")
                logging.debug(f"Hypothesis details: {new_hypothesis}")
        
        logging.debug(f"Generated {len(new_hypotheses)} new hypotheses")
        
        # Prune hypotheses
        new_hypotheses = self.prune_hypotheses(new_hypotheses)
        
        # Merge similar hypotheses
        new_hypotheses = self.merge_hypotheses(new_hypotheses)
        
        # Normalize weights
        total_weight = sum(h['weight'] for h in new_hypotheses)
        for h in new_hypotheses:
            h['weight'] /= total_weight
        
        # Keep only top hypotheses
        self.hypotheses = sorted(new_hypotheses, key=lambda h: h['weight'], reverse=True)[:self.max_hypotheses]
        logging.info(f"Update completed. Kept {len(self.hypotheses)} hypotheses")
        logging.debug(f"Top 5 hypotheses: {self.hypotheses[:5]}")

    def prune_hypotheses(self, hypotheses):
        max_weight = max(h['weight'] for h in hypotheses)
        pruned = [h for h in hypotheses if h['weight'] > max_weight * self.pruning_threshold]
        logging.debug(f"Pruned hypotheses from {len(hypotheses)} to {len(pruned)}")
        return pruned

    def merge_hypotheses(self, hypotheses):
        merged = []
        for h in hypotheses:
            merged_h = next((m for m in merged if self.are_hypotheses_similar(h, m)), None)
            if merged_h is None:
                merged.append(h)
            else:
                merged_h['weight'] += h['weight']
        logging.debug(f"Merged hypotheses from {len(hypotheses)} to {len(merged)}")
        return merged

    def are_hypotheses_similar(self, h1, h2):
        if len(h1['tracks']) != len(h2['tracks']):
            return False
        for t1, t2 in zip(h1['tracks'], h2['tracks']):
            if np.linalg.norm(t1['state'] - t2['state']) > 1e-3:
                return False
        return True

    def get_best_hypothesis(self):
        best = max(self.hypotheses, key=lambda h: h['weight'])
        logging.debug(f"Best hypothesis: {best}")
        return best