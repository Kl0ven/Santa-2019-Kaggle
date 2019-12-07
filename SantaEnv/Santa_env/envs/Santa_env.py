import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym import spaces
import pandas as pd
from numba import njit

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

class Santa_env(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(Santa_env, self).__init__()
		self.state = np.random.randint(0, 100, size=5000)
		self.counter = 0
		self.done = 0
		self.add = [0, 0]
		self.reward = 0
		# Actions of the format Buy x%, Sell x%, Hold, etc.
		self.action_space = spaces.Box(
			low=0, high=1, shape=(2,), dtype=np.float16)

		# Prices contains the OHCL values for the last five prices
		self.observation_space = spaces.Box(
			low=0, high=1, shape=(5000,), dtype=np.float16)

		load_dataset()

	def load_dataset():
		fpath = './family_data.csv'
		self.data = pd.read_csv(fpath, index_col='family_id')
		self.family_size = self.data.n_people.values
		self.days_array = np.arange(N_DAYS, 0, -1)
		self.choice_dict = self.data.loc[:, 'choice_0': 'choice_9'].T.to_dict()

		self.choice_array_num = np.full((self.data.shape[0], N_DAYS + 1), -1)

		for i, choice in enumerate(self.data.loc[:, 'choice_0': 'choice_9'].values):
			for d, day in enumerate(choice):
				self.choice_array_num[i, day] = d

		self.penalties_array = np.array([
			[
				0,
				50,
				50 + 9 * n,
				100 + 9 * n,
				200 + 9 * n,
				200 + 18 * n,
				300 + 18 * n,
				300 + 36 * n,
				400 + 36 * n,
				500 + 36 * n + 199 * n,
				500 + 36 * n + 398 * n
			]
			for n in range(self.family_size.max() + 1)
		])

	def step(self, action):
		if self.done == 1:
			return [self.state, self.reward, self.done, self.add]
		# print(action)

		return [self.state, self.reward, self.done, self.add]

	def reset(self):
		self.state = np.random.randint(0, 100, size=5000)
		self.counter = 0
		self.done = 0
		self.add = [0, 0]
		self.reward = 0
		return self.state


	def render(self, mode='human', close=False):
		pass

	@njit
	def cost_function(prediction, penalties_array, family_size, days):
		penalty = 0

		# We'll use this to count the number of people scheduled each day
		daily_occupancy = np.zeros((len(days)+1))
		N = family_size.shape[0]

		# Looping over each family; d is the day, n is size of that family,
		# and choice is their top choices
		for i in range(N):
			# add the family member count to the daily occupancy
			n = family_size[i]
			d = prediction[i]
			choice = choice_array_num[i]

			daily_occupancy[d] += n

			# Calculate the penalty for not getting top preference
			penalty += penalties_array[n, choice[d]]

		# for each date, check total occupancy
		#  (using soft constraints instead of hard constraints)
		relevant_occupancy = daily_occupancy[1:]
		incorrect_occupancy = np.any(
			(relevant_occupancy > MAX_OCCUPANCY) |
			(relevant_occupancy < MIN_OCCUPANCY)
		)

		if incorrect_occupancy:
			penalty += 100000000

		# Calculate the accounting cost
		# The first day (day 100) is treated special
		init_occupancy = daily_occupancy[days[0]]
		accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)
		# using the max function because the soft constraints might allow occupancy to dip below 125
		accounting_cost = max(0, accounting_cost)

		# Loop over the rest of the days, keeping track of previous count
		yesterday_count = init_occupancy
		for day in days[1:]:
			today_count = daily_occupancy[day]
			diff = np.abs(today_count - yesterday_count)
			accounting_cost += max(0, (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0))
			yesterday_count = today_count

		penalty += accounting_cost

		return penalty
