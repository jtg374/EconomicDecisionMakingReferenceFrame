from __future__ import division
from typing import Union

from psychrnn.tasks.task import Task
import numpy as np

class DelayedAndSimultaneousChoice(Task):
    """Delayed economic decision making task. Ballesta & Padoa-Schioppa 2019
    Offer input restricted to two juice channels

    [TODO] [simple description]

    Takes two channels of taste inputs (A or B) and one or two channels of quantity input  (:attr:`N_in` = ?).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2).

    [TODO] [ref to task]

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        [TODO] [other parameters]
    """
    ###[TODO!!!!] earilier contexual choice
    def __init__(self, dt, tau, T, N_batch=None,
        onset_time = 500, stim_duration_1 = 500, InterOffer_duration = 500, stim_duration_2 = 500, 
        wait_duration = 0, target_delay_duration = 200, respond_duration=200,offerMode='seq',outputMode='juice',offer_pairs=None,N_trials_per_condition=None,ind_point=1.7):
        N_stim = 2
        N_target = 2+2+2
        N_fixation = 1
        N_in = N_stim+N_target+N_fixation
        
        if offer_pairs is not None:
            self.offer_pairs = offer_pairs
        else:
            self.offer_pairs = [(0, 1), 
                            (0, 2), 
                            (1, 0), 
                            (1, 3), 
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6), 
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)] # offer quantity pairs to select from

        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning('fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_offerPair = len(self.offer_pairs)
            if outputMode=='both':
                N_batch = 4*N_offerPair*N_trials_per_condition
            else:
                N_batch = 2*N_offerPair*N_trials_per_condition
        self.N_trials_per_condition = N_trials_per_condition

        super(DelayedAndSimultaneousChoice,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.target_delay_duration = target_delay_duration
        self.respond_duration = respond_duration 
        self.outputMode = outputMode
        self.offerMode = offerMode

        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.spatial_options = ['AB', 'BA'] # juice location to select from
        self.offerPrest_options = ['seq','simul'] # offer present sequentially or simultaneously
        self.choiceFrame_options = ['juice','order'] # target represent juice A/B or order 1st/2nd
        # behavior parameter
        self.ind_point = ind_point #2*np.random.gamma(20,0.05) # indifference point
        self.a1_choice = 13 #15*np.random.gamma(100,0.01) # related to steepness


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding
    
    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        InterOffer_duration     =self.InterOffer_duration
        stim_duration_2         =self.stim_duration_2
        wait_duration           =self.wait_duration
        target_delay_duration   =self.target_delay_duration
        respond_duration        =self.respond_duration 
        outputMode              =self.outputMode
        offerMode               =self.offerMode

        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + wait_duration
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration + target_delay_duration
        params['end'] = params['fixation_offset'] + respond_duration

        params['stim_noise'] = 0.01

        if offerMode == 'both':
            offerPrest = np.random.choice(self.offerPrest_options)
        elif offerMode == 'seq' or offerMode=='simul':
            offerPrest = offerMode

        if offerPrest=='simul':
            params['stimulus_2_onset'] = params['stimulus_2_onset'] - stim_duration_1 - InterOffer_duration
            params['stimulus_2_offset'] = params['stimulus_2_offset'] - stim_duration_1 - InterOffer_duration
            params['target_onset'] = params['target_onset'] - stim_duration_1 - InterOffer_duration
            params['fixation_offset'] = params['fixation_offset'] - stim_duration_1 - InterOffer_duration
            params['end'] = params['fixation_offset'] + respond_duration

        if self.N_trials_per_condition is None:
            offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
            seqAB = np.random.choice(self.seq_options)
            if outputMode == 'both':
                choiceFrame = np.random.choice(self.choiceFrame_options)
            elif outputMode == 'juice' or outputMode == 'order':
                choiceFrame = outputMode

        else:
            if outputMode == 'both':
                offer_pair = self.offer_pairs[int(trial/4/self.N_trials_per_condition)]
                subtrial = np.mod(trial,4*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition/2)]
                subtrial = np.mod(subtrial,2*self.N_trials_per_condition)
                choiceFrame = self.choiceFrame_options[int(subtrial/self.N_trials_per_condition)]
            else:
                offer_pair = self.offer_pairs[int(trial/2/self.N_trials_per_condition)]
                subtrial = np.mod(trial,2*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition)]
                choiceFrame = outputMode
        locAB = np.random.choice(self.spatial_options)

        chooseB = self._choice_stochastic(*offer_pair)

        if choiceFrame == 'juice':
            if locAB == 'AB':
                choice = chooseB # choice in order reference frame (saccade)
            else:
                choice = 1-chooseB
        elif choiceFrame == 'order':
            if seqAB == 'AB':
                choice12 = chooseB
            else:
                choice12 = 1-chooseB
            if locAB == 'AB':
                loc12 = '12'
                choice = choice12
            else:
                loc12 = '21'
                choice = 1-choice12
            
        params['qA'],params['qB'] = offer_pair
        params['choiceFrame'] = choiceFrame
        params['offerPrest']  = offerPrest
        params['seqAB'] = seqAB
        params['locAB'] = locAB if choiceFrame=='juice' else loc12
        params['choice'] = choice
        params['chooseB'] = chooseB

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1_onset=   params['stimulus_1_onset']
        stimulus_1_offset=  params['stimulus_1_offset'] 
        stimulus_2_onset=   params['stimulus_2_onset'] 
        stimulus_2_offset=  params['stimulus_2_offset'] 
        target_onset=       params['target_onset'] 
        fixation_offset=    params['fixation_offset']
        end=                params['end'] 

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        locAB = params['locAB']
        choice = params['choice']
        chooseB = params['chooseB']
        choiceFrame = params['choiceFrame']

        # offer periods
        if seqAB == 'AB':
            c1,c2=(0,1) # stim channel [1A,2B]
            q1, q2 = offer_pair
            q1 = self._range_norm_A(q1)
            q2 = self._range_norm_B(q2)
        else:
            c1,c2=(1,0) # stim channel [1B 2A]
            q2, q1 = offer_pair
            q1 = self._range_norm_B(q1)
            q2 = self._range_norm_A(q2)
       
        # target period
        if choiceFrame == 'juice':
            if locAB == 'AB': # 
                tg1,tg2 = (2,5) # target channels [*leftA,leftB,rightA,*rightB]
            else:
                tg1,tg2 = (3,4) # target channels [leftA,*leftB,,*rightA,rightB]
        elif choiceFrame == 'order':
            loc12 = locAB
            if loc12 == '12':
                tg=6
            else:
                tg=7                


        # ----------------------------------
        # Compute values
        # ----------------------------------
            
        if stimulus_1_onset <= t < stimulus_1_offset:
            x_t[c1] += q1

        if stimulus_2_onset <= t < stimulus_2_offset:
            x_t[c2] += q2

        if target_onset <=t:
            if choiceFrame == 'juice':
                x_t[tg1] +=1
                x_t[tg2] +=1
            elif choiceFrame == 'order':
                x_t[tg] +=1

        if t<fixation_offset:
            x_t[8] +=1
        elif t<end:
            y_t[choice] = self.hi 
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))

class DelayedEconomicDecision_KuntanInput(Task):
    """Delayed economic decision making task. Ballesta & Padoa-Schioppa 2019
    Offer input restricted to two juice channels

    [TODO] [simple description]

    Takes two channels of taste inputs (A or B) and one or two channels of quantity input  (:attr:`N_in` = ?).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2).

    [TODO] [ref to task]

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        [TODO] [other parameters]
    """

    def __init__(self, dt, tau, T, N_batch,
        onset_time = 500, stim_duration_1 = 500, InterOffer_duration = 500, stim_duration_2 = 500, 
        wait_duration = 0, target_delay_duration = 200, respond_duration=200):
        N_stim = 2+2
        N_target = 2
        N_fixation = 1
        N_in = N_stim+N_target+N_fixation
        super(DelayedEconomicDecision_KuntanInput,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.target_delay_duration = target_delay_duration
        self.respond_duration = respond_duration 

        self.offer_pairs = [(0, 1), 
                            (0, 2), 
                            (1, 0), 
                            (1, 3), 
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6), 
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)] # offer quantity pairs to select from
        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.spatial_options = ['AB', 'BA'] # juice location to select from
        # behavior parameter
        self.ind_point = 1.7 #2*np.random.gamma(20,0.05) # indifference point
        self.a1_choice = 13 #15*np.random.gamma(100,0.01) # related to steepness


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding
    
    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        InterOffer_duration     =self.InterOffer_duration
        stim_duration_2         =self.stim_duration_2
        wait_duration           =self.wait_duration
        target_delay_duration   =self.target_delay_duration
        respond_duration        =self.respond_duration 
        
        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + wait_duration
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration + target_delay_duration
        params['end'] = params['fixation_offset'] + respond_duration

        params['stim_noise'] = 0.01

        offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
        seqAB = np.random.choice(self.seq_options)
        locAB = np.random.choice(self.spatial_options)

        chooseB = self._choice_stochastic(*offer_pair)
        if locAB == 'AB':
            choice = chooseB # choice in order reference frame (saccade)
        else:
            choice = 1-chooseB

            
        params['qA'],params['qB'] = offer_pair
        params['seqAB'] = seqAB
        params['locAB'] = locAB
        params['choice'] = choice
        params['chooseB'] = chooseB

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1_onset=   params['stimulus_1_onset']
        stimulus_1_offset=  params['stimulus_1_offset'] 
        stimulus_2_onset=   params['stimulus_2_onset'] 
        stimulus_2_offset=  params['stimulus_2_offset'] 
        target_onset=       params['target_onset'] 
        fixation_offset=    params['fixation_offset']
        end=                params['end'] 

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        locAB = params['locAB']
        choice = params['choice']
        chooseB = params['chooseB']

        # 4 stim channels : leftA, leftB, rightA, rightB
        if locAB == 'AB':
            if seqAB == 'AB':
                c1,c2=(0,3) 
                q1, q2 = offer_pair
                q1 = self._range_norm_A(q1)
                q2 = self._range_norm_B(q2)
            else:
                c2,c1=(0,3) 
                q2, q1 = offer_pair
                q2 = self._range_norm_A(q1)
                q1 = self._range_norm_B(q2)
        else:
            if seqAB == 'AB':
                c1,c2=(1,2) 
                q1, q2 = offer_pair
                q1 = self._range_norm_A(q1)
                q2 = self._range_norm_B(q2)
            else:
                c2,c1=(1,2) 
                q2, q1 = offer_pair
                q2 = self._range_norm_A(q1)
                q1 = self._range_norm_B(q2)

        tg1,tg2 = (4,5) # target channels are of neutral color in Kuntan task


        # ----------------------------------
        # Compute values
        # ----------------------------------
            
        if stimulus_1_onset <= t < stimulus_1_offset:
            x_t[c1] += q1

        if stimulus_2_onset <= t < stimulus_2_offset:
            x_t[c2] += q2

        if target_onset <=t:
            x_t[tg1] +=1
            x_t[tg2] +=1

        if t<fixation_offset:
            x_t[6] +=1
        else:
            y_t[choice] = self.hi 
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))

class DelayedEconomicDecision_Perturbation(Task):
    """Delayed economic decision making task. Ballesta & Padoa-Schioppa 2019
    Offer input restricted to two juice channels

    [TODO] [simple description]

    Takes two channels of taste inputs (A or B) and one or two channels of quantity input  (:attr:`N_in` = ?).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2).

    [TODO] [ref to task]

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        [TODO] [other parameters]
    """
    def __init__(self, dt, tau, T, N_batch=None,
        onset_time = 500, stim_duration_1 = 500, InterOffer_duration = 500, stim_duration_2 = 500, 
        early_conxt=False,
        wait_duration = 0, target_delay_duration = 200, respond_duration=200,outputMode='both',juiceTrialProp=0.5,offer_pairs=None,N_trials_per_condition=None,ind_point=1.7,
        perturb_time=1200,perturb_duration=50,perturb_juice='A',perturb_quantity=4):
        N_stim = 2
        N_target = 2+2+2
        N_fixation = 2 if early_conxt else 1
        N_in = N_stim+N_target+N_fixation
        
        self._early_conxt = early_conxt
        self._input_names = ['qA_norm','qB_norm','A_','B_','_A','_B','12','21','fixation']
        if early_conxt:
            self._input_names = ['qA_norm','qB_norm','A_','B_','_A','_B','12','21','fix_juice','fix_order']
        self.output_names = ['left','right']

        if offer_pairs is not None:
            self.offer_pairs = offer_pairs
        else:
            self.offer_pairs = [(0, 1), 
                            (0, 2), 
                            (1, 0), 
                            (1, 3), 
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6), 
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)] # offer quantity pairs to select from

        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning('fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_offerPair = len(self.offer_pairs)
            if outputMode=='both':
                N_batch = 4*N_offerPair*N_trials_per_condition
            else:
                N_batch = 2*N_offerPair*N_trials_per_condition
        self.N_trials_per_condition = N_trials_per_condition

        super(DelayedEconomicDecision_Perturbation,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.target_delay_duration = target_delay_duration
        self.respond_duration = respond_duration 
        self.outputMode = outputMode
        if outputMode == 'both':    self.juiceTrialProp=juiceTrialProp

        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.spatial_options = ['AB', 'BA'] # juice location to select from
        self.choiceFrame_options = ['juice','order'] # target represent juice A/B or order 1st/2nd
        # behavior parameter
        self.ind_point = ind_point #2*np.random.gamma(20,0.05) # indifference point
        self.a1_choice = 13 #15*np.random.gamma(100,0.01) # related to steepness


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding
        
        self.perturb_time=perturb_time
        self.perturb_duration=perturb_duration
        self.perturb_juice=perturb_juice
        self.perturb_quantity=perturb_quantity

    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        InterOffer_duration     =self.InterOffer_duration
        stim_duration_2         =self.stim_duration_2
        wait_duration           =self.wait_duration
        target_delay_duration   =self.target_delay_duration
        respond_duration        =self.respond_duration 
        outputMode              =self.outputMode

        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + wait_duration
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration + target_delay_duration
        params['end'] = params['fixation_offset'] + respond_duration

        params['stim_noise'] = 0.01
        if self.N_trials_per_condition is None:
            offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
            seqAB = np.random.choice(self.seq_options)
            if outputMode == 'both':
                if self.juiceTrialProp==0.5:
                    choiceFrame = np.random.choice(self.choiceFrame_options)
                else:
                    choiceFrame = np.random.choice(self.choiceFrame_options,p=[self.juiceTrialProp,1-self.juiceTrialProp])
            elif outputMode == 'juice' or outputMode == 'order':
                choiceFrame = outputMode
        else: # for testing, all condition equal trials, no juice trial proportion. 
            if outputMode == 'both':
                offer_pair = self.offer_pairs[int(trial/4/self.N_trials_per_condition)]
                subtrial = np.mod(trial,4*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition/2)]
                subtrial = np.mod(subtrial,2*self.N_trials_per_condition)
                choiceFrame = self.choiceFrame_options[int(subtrial/self.N_trials_per_condition)]
            else:
                offer_pair = self.offer_pairs[int(trial/2/self.N_trials_per_condition)]
                subtrial = np.mod(trial,2*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition)]
                choiceFrame = outputMode
        locAB = np.random.choice(self.spatial_options)

        chooseB = self._choice_stochastic(*offer_pair)

        if choiceFrame == 'juice':
            if locAB == 'AB':
                choice = chooseB # choice in order reference frame (saccade)
            else:
                choice = 1-chooseB
        elif choiceFrame == 'order':
            if seqAB == 'AB':
                choice12 = chooseB
            else:
                choice12 = 1-chooseB
            if locAB == 'AB':
                loc12 = '12'
                choice = choice12
            else:
                loc12 = '21'
                choice = 1-choice12
            
        params['qA'],params['qB'] = offer_pair
        params['choiceFrame'] = choiceFrame
        params['seqAB'] = seqAB
        if choiceFrame=='juice' :
            params['locAB'] = locAB 
        else:
            params['locAB'] = loc12 # for compatibility 
            params['loc12'] = loc12
        params['choice'] = choice
        params['chooseB'] = chooseB


        params['perturb_time']=self.perturb_time
        params['perturb_duration']=self.perturb_duration
        params['perturb_juice']=self.perturb_juice
        params['perturb_quantity']=self.perturb_quantity

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1_onset=   params['stimulus_1_onset']
        stimulus_1_offset=  params['stimulus_1_offset'] 
        stimulus_2_onset=   params['stimulus_2_onset'] 
        stimulus_2_offset=  params['stimulus_2_offset'] 
        target_onset=       params['target_onset'] 
        fixation_offset=    params['fixation_offset']
        end=                params['end'] 

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        locAB = params['locAB']
        choice = params['choice']
        chooseB = params['chooseB']
        choiceFrame = params['choiceFrame']

        # offer periods
        if seqAB == 'AB':
            c1,c2=(0,1) # stim channel [1A,2B]
            q1, q2 = offer_pair
            q1 = self._range_norm_A(q1)
            q2 = self._range_norm_B(q2)
        else:
            c1,c2=(1,0) # stim channel [1B 2A]
            q2, q1 = offer_pair
            q1 = self._range_norm_B(q1)
            q2 = self._range_norm_A(q2)
       
        if self.perturb_juice == 'A':
            cPrtb = 0
            qPrtb = self._range_norm_A(self.perturb_quantity)
        elif self.perturb_juice == 'B':
            cPrtb = 1
            qPrtb = self._range_norm_B(self.perturb_quantity)

        # target period
        if choiceFrame == 'juice':
            fix_channel = 8
            if locAB == 'AB': # 
                tg1,tg2 = (2,5) # target channels [*leftA,leftB,rightA,*rightB]
            else:
                tg1,tg2 = (3,4) # target channels [leftA,*leftB,,*rightA,rightB]
        elif choiceFrame == 'order':
            fix_channel = 9 if self._early_conxt else 8
            loc12 = locAB
            if loc12 == '12':
                tg=6
            else:
                tg=7                


        # ----------------------------------
        # Compute values
        # ----------------------------------
            
        if stimulus_1_onset <= t < stimulus_1_offset:
            x_t[c1] += q1

        if stimulus_2_onset <= t < stimulus_2_offset:
            x_t[c2] += q2

        if self.perturb_time <= t < (self.perturb_time+self.perturb_duration):
            x_t[cPrtb]=qPrtb

        if target_onset <=t:
            if choiceFrame == 'juice':
                x_t[tg1] +=1
                x_t[tg2] +=1
            elif choiceFrame == 'order':
                x_t[tg] +=1

        if t<fixation_offset:
            x_t[fix_channel] +=1
        else:
            y_t[choice] = self.hi 
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))

class DelayedEconomicDecision_AlternateOutput(Task):
    """Delayed economic decision making task. Ballesta & Padoa-Schioppa 2019
    Offer input restricted to two juice channels

    [TODO] [simple description]

    Takes two channels of taste inputs (A or B) and one or two channels of quantity input  (:attr:`N_in` = ?).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2).

    [TODO] [ref to task]

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        [TODO] [other parameters]
    """
    def __init__(self, dt, tau, T, N_batch=None,
        onset_time = 500, stim_duration_1 = 500, InterOffer_duration: Union[range,int] = 500, stim_duration_2 = 500, 
        early_conxt=False,
        wait_duration = 0, target_delay_duration = 200, respond_duration=200,outputMode='both',juiceTrialProp=0.5,offer_pairs=None,N_trials_per_condition=None,ind_point=1.7):
        N_stim = 2
        N_target = 2+2+2
        N_fixation = 2 if early_conxt else 1
        N_in = N_stim+N_target+N_fixation
        
        self._early_conxt = early_conxt
        self._input_names = ['qA_norm','qB_norm','A_','B_','_A','_B','12','21','fixation']
        if early_conxt:
            self._input_names = ['qA_norm','qB_norm','A_','B_','_A','_B','12','21','fix_juice','fix_order']
        self.output_names = ['left','right']

        if offer_pairs is not None:
            self.offer_pairs = offer_pairs
        else:
            self.offer_pairs = [(0, 1), 
                            (0, 2), 
                            (1, 0), 
                            (1, 3), 
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6), 
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)] # offer quantity pairs to select from

        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning('fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_offerPair = len(self.offer_pairs)
            if outputMode=='both':
                N_batch = 4*N_offerPair*N_trials_per_condition
            else:
                N_batch = 2*N_offerPair*N_trials_per_condition
        self.N_trials_per_condition = N_trials_per_condition

        super(DelayedEconomicDecision_AlternateOutput,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.target_delay_duration = target_delay_duration
        self.respond_duration = respond_duration 
        self.outputMode = outputMode
        if outputMode == 'both':    self.juiceTrialProp=juiceTrialProp

        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.spatial_options = ['AB', 'BA'] # juice location to select from
        self.choiceFrame_options = ['juice','order'] # target represent juice A/B or order 1st/2nd
        # behavior parameter
        self.ind_point = ind_point #2*np.random.gamma(20,0.05) # indifference point
        self.a1_choice = 13 #15*np.random.gamma(100,0.01) # related to steepness


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding
    
    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        InterOffer_duration     =self.InterOffer_duration
        if type(InterOffer_duration) in (range,list,tuple):
            InterOffer_duration = np.random.choice(InterOffer_duration)
        stim_duration_2         =self.stim_duration_2
        wait_duration           =self.wait_duration
        target_delay_duration   =self.target_delay_duration
        respond_duration        =self.respond_duration 
        outputMode              =self.outputMode

        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + wait_duration
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration + target_delay_duration
        params['end'] = params['fixation_offset'] + respond_duration

        params['stim_noise'] = 0.01
        if self.N_trials_per_condition is None:
            offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
            seqAB = np.random.choice(self.seq_options)
            if outputMode == 'both':
                if self.juiceTrialProp==0.5:
                    choiceFrame = np.random.choice(self.choiceFrame_options)
                else:
                    choiceFrame = np.random.choice(self.choiceFrame_options,p=[self.juiceTrialProp,1-self.juiceTrialProp])
            elif outputMode == 'juice' or outputMode == 'order':
                choiceFrame = outputMode
        else: # for testing, all condition equal trials, no juice trial proportion. 
            if outputMode == 'both':
                offer_pair = self.offer_pairs[int(trial/4/self.N_trials_per_condition)]
                subtrial = np.mod(trial,4*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition/2)]
                subtrial = np.mod(subtrial,2*self.N_trials_per_condition)
                choiceFrame = self.choiceFrame_options[int(subtrial/self.N_trials_per_condition)]
            else:
                offer_pair = self.offer_pairs[int(trial/2/self.N_trials_per_condition)]
                subtrial = np.mod(trial,2*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition)]
                choiceFrame = outputMode
        locAB = np.random.choice(self.spatial_options)

        chooseB = self._choice_stochastic(*offer_pair)

        if choiceFrame == 'juice':
            if locAB == 'AB':
                choice = chooseB # choice in order reference frame (saccade)
            else:
                choice = 1-chooseB
        elif choiceFrame == 'order':
            if seqAB == 'AB':
                choice12 = chooseB
            else:
                choice12 = 1-chooseB
            if locAB == 'AB':
                loc12 = '12'
                choice = choice12
            else:
                loc12 = '21'
                choice = 1-choice12
            
        params['qA'],params['qB'] = offer_pair
        params['choiceFrame'] = choiceFrame
        params['seqAB'] = seqAB
        if choiceFrame=='juice' :
            params['locAB'] = locAB 
        else:
            params['locAB'] = loc12 # for compatibility 
            params['loc12'] = loc12
        params['choice'] = choice
        params['chooseB'] = chooseB

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1_onset=   params['stimulus_1_onset']
        stimulus_1_offset=  params['stimulus_1_offset'] 
        stimulus_2_onset=   params['stimulus_2_onset'] 
        stimulus_2_offset=  params['stimulus_2_offset'] 
        target_onset=       params['target_onset'] 
        fixation_offset=    params['fixation_offset']
        end=                params['end'] 

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        locAB = params['locAB']
        choice = params['choice']
        chooseB = params['chooseB']
        choiceFrame = params['choiceFrame']

        # offer periods
        if seqAB == 'AB':
            c1,c2=(0,1) # stim channel [1A,2B]
            q1, q2 = offer_pair
            q1 = self._range_norm_A(q1)
            q2 = self._range_norm_B(q2)
        else:
            c1,c2=(1,0) # stim channel [1B 2A]
            q2, q1 = offer_pair
            q1 = self._range_norm_B(q1)
            q2 = self._range_norm_A(q2)
       
        # target period
        if choiceFrame == 'juice':
            fix_channel = 8
            if locAB == 'AB': # 
                tg1,tg2 = (2,5) # target channels [*leftA,leftB,rightA,*rightB]
            else:
                tg1,tg2 = (3,4) # target channels [leftA,*leftB,,*rightA,rightB]
        elif choiceFrame == 'order':
            fix_channel = 9 if self._early_conxt else 8
            loc12 = locAB
            if loc12 == '12':
                tg=6
            else:
                tg=7                


        # ----------------------------------
        # Compute values
        # ----------------------------------
            
        if stimulus_1_onset <= t < stimulus_1_offset:
            x_t[c1] += q1

        if stimulus_2_onset <= t < stimulus_2_offset:
            x_t[c2] += q2

        if target_onset <=t:
            if choiceFrame == 'juice':
                x_t[tg1] +=1
                x_t[tg2] +=1
            elif choiceFrame == 'order':
                x_t[tg] +=1

        if t<fixation_offset:
            x_t[fix_channel] +=1
        else:
            y_t[choice] = self.hi 
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))

class DelayedEconomicDecision_labelLine(Task):
    """Delayed economic decision making task. Ballesta & Padoa-Schioppa 2019
    Offer input restricted to two juice channels

    [TODO] [simple description]

    Takes two channels of taste inputs (A or B) and one or two channels of quantity input  (:attr:`N_in` = ?).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2).

    [TODO] [ref to task]

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        [TODO] [other parameters]
    """

    def __init__(self, dt, tau, T, N_batch,
        onset_time = 500, stim_duration_1 = 500, InterOffer_duration = 500, stim_duration_2 = 500, 
        wait_duration = 0, target_delay_duration = 200, respond_duration=200):
        N_stim = 2
        N_target = 2+2
        N_fixation = 1
        N_in = N_stim+N_target+N_fixation
        super(DelayedEconomicDecision_labelLine,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.target_delay_duration = target_delay_duration
        self.respond_duration = respond_duration 

        self.offer_pairs = [(0, 1), 
                            (0, 2), 
                            (1, 0), 
                            (1, 3), 
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6), 
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)] # offer quantity pairs to select from
        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.spatial_options = ['AB', 'BA'] # juice location to select from
        # behavior parameter
        self.ind_point = 1.7 #2*np.random.gamma(20,0.05) # indifference point
        self.a1_choice = 13 #15*np.random.gamma(100,0.01) # related to steepness


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding
    
    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        InterOffer_duration     =self.InterOffer_duration
        stim_duration_2         =self.stim_duration_2
        wait_duration           =self.wait_duration
        target_delay_duration   =self.target_delay_duration
        respond_duration        =self.respond_duration 
        
        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + wait_duration
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration + target_delay_duration
        params['end'] = params['fixation_offset'] + respond_duration

        params['stim_noise'] = 0.01

        offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
        seqAB = np.random.choice(self.seq_options)
        locAB = np.random.choice(self.spatial_options)

        chooseB = self._choice_stochastic(*offer_pair)
        if locAB == 'AB':
            choice = chooseB # choice in order reference frame (saccade)
        else:
            choice = 1-chooseB

            
        params['qA'],params['qB'] = offer_pair
        params['seqAB'] = seqAB
        params['locAB'] = locAB
        params['choice'] = choice
        params['chooseB'] = chooseB

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1_onset=   params['stimulus_1_onset']
        stimulus_1_offset=  params['stimulus_1_offset'] 
        stimulus_2_onset=   params['stimulus_2_onset'] 
        stimulus_2_offset=  params['stimulus_2_offset'] 
        target_onset=       params['target_onset'] 
        fixation_offset=    params['fixation_offset']
        end=                params['end'] 

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        locAB = params['locAB']
        choice = params['choice']
        chooseB = params['chooseB']

        if seqAB == 'AB':
            c1,c2=(0,1) # stim channel [1A,2B]
            q1, q2 = offer_pair
            q1 = self._range_norm_A(q1)
            q2 = self._range_norm_B(q2)
        else:
            c1,c2=(1,0) # stim channel [1B 2A]
            q2, q1 = offer_pair
            q1 = self._range_norm_B(q1)
            q2 = self._range_norm_A(q2)

        if locAB == 'AB': # 
            tg1,tg2 = (2,5) # target channels [*leftA,leftB,rightA,*rightB]
        else:
            tg1,tg2 = (3,4) # target channels [leftA,*leftB,,*rightA,rightB]


        # ----------------------------------
        # Compute values
        # ----------------------------------
            
        if stimulus_1_onset <= t < stimulus_1_offset:
            x_t[c1] += q1

        if stimulus_2_onset <= t < stimulus_2_offset:
            x_t[c2] += q2

        if target_onset <=t:
            x_t[tg1] +=1
            x_t[tg2] +=1

        if t<fixation_offset:
            x_t[6] +=1
        else:
            y_t[choice] = self.hi 
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))

class DelayedEconomicDecision(Task):
    """Delayed economic decision making task. 

    [TODO] [simple description]

    Takes two channels of taste inputs (A or B) and one or two channels of quantity input  (:attr:`N_in` = ?).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2).

    [TODO] [ref to task]

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        [TODO] [other parameters]
    """

    def __init__(self, dt, tau, T, N_batch, spatial_stim = False,
        onset_time = 500, stim_duration_1 = 500, delay1_duration = 1000, stim_duration_2 = 500, 
        delay2_duration = 0, respond_duration=500):
        N_color = 2
        N_quantity = 2 if spatial_stim else 1
        N_respond = 1
        N_in = N_color+N_quantity+N_respond
        super(DelayedEconomicDecision,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.spatial_stim = spatial_stim
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.delay_duration = delay1_duration
        self.stim_duration_2 = stim_duration_2
        self.delay2_duration = delay2_duration
        self.respond_duration = respond_duration 

        self.offer_pairs = [(0, 1), 
                            (0, 2), 
                            (1, 0), 
                            (1, 3), 
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6), 
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)] # offer quantity pairs to select from
        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.spatial_options = ['AB', 'BA'] # juice location to select from
        self.ind_point = 2*np.random.gamma(20,0.05)
        self.a1_choice = 15*np.random.gamma(100,0.01)


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding
    
    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        delay1_duration         =self.delay_duration
        stim_duration_2         =self.stim_duration_2
        delay2_duration         =self.delay2_duration
        respond_duration        =self.respond_duration 
        
        params['stimulus_1'] = onset_time
        params['delay1'] = onset_time + stim_duration_1
        params['stimulus_2'] = onset_time + stim_duration_1 + delay1_duration
        params['delay2'] = onset_time + stim_duration_1 + delay1_duration + stim_duration_2
        params['decision'] = onset_time + stim_duration_1 + delay1_duration + stim_duration_2 + delay2_duration
        params['end'] = onset_time + stim_duration_1 + delay1_duration + stim_duration_2 + delay2_duration + respond_duration

        params['stim_noise'] = 0.01

        offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
        seqAB = np.random.choice(self.seq_options)
        locAB = np.random.choice(self.spatial_options)

        chooseB = self._choice_stochastic(*offer_pair)
        if locAB == 'AB':
            choice = chooseB # choice in spatial reference frame (saccade)
        else:
            choice = 1-chooseB

        params['qA'],params['qB'] = offer_pair
        params['seqAB'] = seqAB
        params['locAB'] = locAB
        params['choice'] = choice
        params['chooseB'] = chooseB

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1 = params['stimulus_1']
        delay1 = params['delay1']
        stimulus_2 = params['stimulus_2']
        delay2 = params['delay2']
        decision = params['decision']
        end = params['end']

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        locAB = params['locAB']
        choice = params['choice']

        spatial_stim = self.spatial_stim

        if seqAB == 'AB':
            c1,c2 = (0,1)
            q1, q2 = offer_pair
            q1 = self._range_norm_A(q1)
            q2 = self._range_norm_B(q2)
        else:
            c1,c2 = (1,0)
            q2, q1 = offer_pair
            q1 = self._range_norm_B(q1)
            q2 = self._range_norm_A(q2)

        if locAB == 'AB':
            s1,s2 = (c1+2,c2+2)
            d = 1
        else:
            s2,s1 = (c1+2,c2+2)
            d=-1

        # ----------------------------------
        # Compute values
        # ----------------------------------

        if spatial_stim:
            if stimulus_1 <= t < delay1:
                x_t[c1] += 1
                x_t[s1] += q1

            if stimulus_2 <= t < delay2:
                x_t[c2] += 1
                x_t[s2] += q2

            if decision <= t < end:
                x_t[4] += d
                y_t[choice] = self.hi
                y_t[1-choice] = self.lo
                mask_t = np.ones(self.N_out)
        else:
            if stimulus_1 <= t < delay1:
                x_t[c1] += 1
                x_t[2] += q1

            if stimulus_2 <= t < delay2:
                x_t[c2] += 1
                x_t[2] += q2

            if decision <= t < end:
                x_t[3] += d
                y_t[choice] = self.hi
                y_t[1-choice] = self.lo
                mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))

class DelayedEconomicDecision_noTarget(Task):
    """Delayed economic decision making task. Ballesta & Padoa-Schioppa 2019
    Respond after seeing two offers. 
    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        [TODO] [other parameters]
    """
    def __init__(self, dt, tau, T, N_batch=None,
        onset_time = 500, stim_duration_1 = 500, InterOffer_duration = 500, stim_duration_2 = 500, 
        wait_duration = 500, respond_duration=200,outputMode='both',offer_pairs=None,N_trials_per_condition=None,ind_point=1.7):
        N_stim = 2
        N_fixation = 2 
        N_in = N_stim+N_fixation
        
        self._input_names = ['qA_norm','qB_norm','fix_juice','fix_order']
        self.output_names = ['1st','2nd'] if outputMode=='order' else ['A','B']

        if offer_pairs is not None:
            self.offer_pairs = offer_pairs
        else:
            self.offer_pairs = [(0, 1), 
                            (0, 2), 
                            (1, 0), 
                            (1, 3), 
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6), 
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)] # offer quantity pairs to select from

        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning('fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_offerPair = len(self.offer_pairs)
            if outputMode=='both':
                N_batch = 4*N_offerPair*N_trials_per_condition
            else:
                N_batch = 2*N_offerPair*N_trials_per_condition
        self.N_trials_per_condition = N_trials_per_condition

        super(DelayedEconomicDecision_noTarget,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.respond_duration = respond_duration 
        self.outputMode = outputMode

        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.choiceFrame_options = ['juice','order'] # target represent juice A/B or order 1st/2nd
        # behavior parameter
        self.ind_point = ind_point #2*np.random.gamma(20,0.05) # indifference point
        self.a1_choice = 13 #15*np.random.gamma(100,0.01) # related to steepness


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding
    
    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        InterOffer_duration     =self.InterOffer_duration
        stim_duration_2         =self.stim_duration_2
        wait_duration           =self.wait_duration
        respond_duration        =self.respond_duration 
        outputMode              =self.outputMode

        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration
        params['end'] = params['fixation_offset'] + respond_duration

        params['stim_noise'] = 0.01
        if self.N_trials_per_condition is None:
            offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
            seqAB = np.random.choice(self.seq_options)
            if outputMode == 'both':
                choiceFrame = np.random.choice(self.choiceFrame_options)
            elif outputMode == 'juice' or outputMode == 'order':
                choiceFrame = outputMode
        else:
            if outputMode == 'both':
                offer_pair = self.offer_pairs[int(trial/4/self.N_trials_per_condition)]
                subtrial = np.mod(trial,4*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition/2)]
                subtrial = np.mod(subtrial,2*self.N_trials_per_condition)
                choiceFrame = self.choiceFrame_options[int(subtrial/self.N_trials_per_condition)]
            else:
                offer_pair = self.offer_pairs[int(trial/2/self.N_trials_per_condition)]
                subtrial = np.mod(trial,2*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition)]
                choiceFrame = outputMode

        chooseB = self._choice_stochastic(*offer_pair)

        if choiceFrame == 'juice':
            choice = chooseB # choice in order reference frame (saccade)
        elif choiceFrame == 'order':
            if seqAB == 'AB':
                choice12 = chooseB
            else:
                choice12 = 1-chooseB
            choice = choice12
            
        params['qA'],params['qB'] = offer_pair
        params['choiceFrame'] = choiceFrame
        params['seqAB'] = seqAB

        params['choice'] = choice
        params['chooseB'] = chooseB

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1_onset=   params['stimulus_1_onset']
        stimulus_1_offset=  params['stimulus_1_offset'] 
        stimulus_2_onset=   params['stimulus_2_onset'] 
        stimulus_2_offset=  params['stimulus_2_offset'] 
        fixation_offset=    params['fixation_offset']
        end=                params['end'] 

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        choice = params['choice']
        chooseB = params['chooseB']
        choiceFrame = params['choiceFrame']

        # fixation signal
        if choiceFrame == self.choiceFrame_options[0]:
            fix_channel = 2 
        elif choiceFrame == self.choiceFrame_options[1]:
            fix_channel = 3

        # offer periods
        if seqAB == 'AB':
            c1,c2=(0,1) # stim channel [1A,2B]
            q1, q2 = offer_pair
            q1 = self._range_norm_A(q1)
            q2 = self._range_norm_B(q2)
        else:
            c1,c2=(1,0) # stim channel [1B 2A]
            q2, q1 = offer_pair
            q1 = self._range_norm_B(q1)
            q2 = self._range_norm_A(q2)
       
           


        # ----------------------------------
        # Compute values
        # ----------------------------------
            
        if stimulus_1_onset <= t < stimulus_1_offset:
            x_t[c1] += q1

        if stimulus_2_onset <= t < stimulus_2_offset:
            x_t[c2] += q2


        if t<fixation_offset:
            x_t[fix_channel] +=1
        else:
            y_t[choice] = self.hi 
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))
