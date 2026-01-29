import numpy as np

class DataSimulation:
    def __init__(self, n_trial, n_ch, n_bins, n_stim, onset_time, stim_effect, base_rate, rw_sigma, rate_max, stim_dur_bins):
        ''' Initialize parameters
        '''
        self.n_trials = n_trial                 # number of trials
        self.n_ch = n_ch                        # number of channels
        self.n_bins = n_bins                    # number of time bins
        self.n_stim = n_stim                    # number of channels to stim each time
        self.onset_time = onset_time            # stimulation onset time bin
        self.stim_effect = stim_effect          # amount of firing rate change
        self.base_rate = base_rate              # base firing rate
        self.rw_sigma = rw_sigma                # random walk step size
        self.rate_max = rate_max                # maximum firing rate
        self.stim_dur_bins = stim_dur_bins      # stimulation duration in bins
        
        self.stim_trials_chs = np.zeros((self.n_trials, self.n_ch))
        self.rates = np.zeros((self.n_trials, self.n_ch, self.n_bins)) # firing rate vector
    
    def generate_stim_trials(self):
        ''' Generate stimulation trials
        '''
        for trial in range(self.n_trials):
            stim_channels = np.random.choice(self.n_ch, self.n_stim, replace=False)
            self.stim_trials_chs[trial, stim_channels] = 1
        return self.stim_trials_chs
        
    def generate_rates(self, add_stim=False):
        ''' Generate rates as bounded random walk
        '''
        initial_rate = np.random.uniform(0, self.rate_max, size=(self.n_trials, self.n_ch))
        self.rates[:, :, 0] = initial_rate

        stim_start = self.onset_time
        stim_end = min(self.onset_time + self.stim_dur_bins, self.n_bins)

        # Bounded random walk
        for bin in range(1, self.n_bins):
            prev_rate = self.rates[:, :, bin - 1].copy()
            if add_stim and (bin == stim_end):
                # Add Gaussian perturbation to rates          
                stim_mask = self.stim_trials_chs  # (trials, ch)
                stim_effect = np.random.normal(self.stim_effect, self.rw_sigma, size=prev_rate.shape) # selected channels for this trial
                prev_rate = np.clip(prev_rate + stim_effect * stim_mask, 0.0, None)

            channels_above_max = prev_rate >= self.rate_max
            step = np.random.normal(0.0, self.rw_sigma, size=prev_rate.shape)
            step[channels_above_max] = np.random.normal(-10, self.rw_sigma, size=step[channels_above_max].shape) # -1 drift down if above max

            self.rates[:, :, bin] = np.clip(prev_rate + step, 0.0, None)
            
        if add_stim and self.stim_dur_bins > 0:
            stim_start = self.onset_time
            stim_end = min(self.onset_time + self.stim_dur_bins, self.n_bins)
            self.rates[:, :, stim_start:stim_end] = np.nan
            
        return self.rates

    def poisson_spikes(self, rates):
        """Generate Poisson spikes for finite rates; keep NaNs as NaNs."""
        spikes = np.full_like(rates, np.nan, dtype=float)
        valid = ~np.isnan(rates)
        spikes[valid] = np.random.poisson(rates[valid])
        return spikes
