import numpy as np

class DataSimulation:
    def __init__(self, n_trial, n_ch, n_bins, n_stim, onset_time, base_rate, stim_effect):
        ''' Initialize parameters
        '''
        self.n_trials = n_trial                 # number of trials
        self.n_ch = n_ch                        # number of channels
        self.n_bins = n_bins                    # number of time bins
        self.n_stim = n_stim                    # number of channels to stim each time
        self.onset_time = onset_time            # stimulation onset time bin
        self.base_rate = base_rate              # base firing rate
        self.stim_effect = stim_effect          # amount of firing rate change
        self.stim_trials = np.zeros((self.n_trials, self.n_ch))

    def generate_stim_trials(self):
        ''' Generate stimulation trials
        '''
        for trial in range(self.n_trials):
            stim_channels = np.random.choice(self.n_ch, self.n_stim, replace=False)
            self.stim_trials[trial, stim_channels] = 1
        return self.stim_trials
        
    def add_stim_perturbation(self, rates):
        ''' Add Gaussian perturbation to rates
        '''
        for trial in range(self.n_trials):
            perturbation = self.stim_effect * self.stim_trials[trial][None, :]
            rates[trial, self.onset_time:, :] += perturbation
        rates_nonzero = np.clip(rates, 0.0, None) # replace this with softplus?
        return rates_nonzero

    def poisson_spikes(self, rates):
        ''' Generate Poisson spikes
        '''
        return np.random.poisson(rates)
