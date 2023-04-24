from gym_anytrading.envs import StocksEnv, TradingEnv
from gym import spaces
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

class Actions(Enum):
    Sell = 0
    Hold = 1
    Buy = 2

# Configure an environment for training - specifically we take StocksEnv
# and customize it according to the problem we're trying to solve
class MyCustomEnv(StocksEnv, TradingEnv):
    # Call the StocksEnv constructor, and on top of that, initialize our own variables
    def __init__(self, df, window_size, frame_bound, MAX_SHARES):
        super().__init__(df, window_size, frame_bound)

        # We override StocksEnv's action space with our own action space, according to
        # the Actions enum class we wrote above
        self.action_space = spaces.Discrete(len(Actions))

        # The total number of shares we currently own
        self.total_shares = 0

        # Maximum number of shares an agent can hold at any given time
        # It keeps margin trading from getting too out of control
        self.MAX_SHARES = MAX_SHARES
        
        # The prices of the shares we bought
        self.prices_of_shares_we_bought = [] 
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # List of actions we've taken over the course of an episode
        self.action_history = []

        # Unrealized gain/loss the agent has at the end of the episode
        self._unrealized_gain = 0
    
    # ====================================================================
    # Override the StocksEnv functions with our own functions
    # ====================================================================
    features_list=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
       'Stock Splits', 'Capital Gains', 'volume_adi', 'volume_obv',
       'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt',
       'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm',
       'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
       'volatility_bbhi', 'volatility_bbli', 'volatility_kcc',
       'volatility_kch', 'volatility_kcl', 'volatility_kcw', 'volatility_kcp',
       'volatility_kchi', 'volatility_kcli', 'volatility_dcl',
       'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp',
       'volatility_atr', 'volatility_ui', 'trend_macd', 'trend_macd_signal',
       'trend_macd_diff', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast',
       'trend_ema_slow', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg',
       'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index', 'trend_dpo',
       'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
       'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
       'trend_psar_down', 'trend_psar_up_indicator',
       'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
       'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
       'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
       'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
       'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
       'others_cr']

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        # Prices over the window size
        prices = self.df.loc[:, 'Close'] #close prices
        prices = prices.to_numpy()[start:end]

        # Features to use as signal: need to adjust
        #Use: momentum_roc: identify overall  percent change in price from one period to the next
        #momentum_stoch_rsi: attuned to a specific security’s historical performance rather than a generalized analysis of price change.
        #momentum_tsi: shows both trend direction and overbought/ oversold conditions
        #momentum_kama: adjusts for noise or volatility
        #volatility_atr: degree of price volatility
        #trend_aroon_ind: identify when trends are likely to go up or down
        #trend_cci: measures the difference between a security’s price change and its average price change. Positive = Strength; Negative =  Weakness
        #'trend_ema_fast' : EMA fast
        #'trend_ema_slow' : EMA slow
        #'trend_kst' : identify major stock market  cycle junctures influenced by the longer and more dominant time spans, in order to better reflect the primary swings of stock market cycle
        #'trend_stc' :  charting indicator that is commonly used to identify market trends and provide buy and sell signals to traders; currency trends accelerate and decelerate in cyclical patterns.
        #'trend_vortex_ind_pos' : consists of two oscillators that capture positive and negative trend movement.
        #'trend_mass_index':  high-low range to identify trend reversals based on range expansions. It identifies range bulges that can foreshadow a reversal of the current trend
        feature_space = ['volatility_atr', 'volume_obv', 'momentum_stoch_rsi', 'trend_vortex_ind_pos',  'trend_macd_diff']
        signal_features = self.df.loc[:, feature_space] #,'volatility_atr', 'trend_macd_diff'
        #TODO cash at hand value; incorporate other features to set constraints amounts to buy/sell 
        signal_features = signal_features.to_numpy()[start:end]
        print("signal_features =", signal_features.shape)
        return prices, signal_features
    
    def _calculate_reward(self, action):
        # Initialize a variable for the reward the agent receives from taking this action
        step_reward = 0

        if action == Actions.Hold.value:
            self.action_history.append(action)
            return step_reward
        
        elif action == Actions.Buy.value and self.total_shares < self.MAX_SHARES:
            self.action_history.append(action)

            current_price = self.prices[self._current_tick]
            self.prices_of_shares_we_bought.append(current_price)
            self.total_shares += 1
            self.buy_count += 1
            
            # We don't reward buying, because we don't know yet if we'll profit from this
            # trade or not. If we rewarded buying, then the agent might keep buying forever,
            # and never selling.
            return step_reward
        
        elif action == Actions.Sell.value and self.total_shares > 0:
            self.action_history.append(action)
            
            current_price = self.prices[self._current_tick]
            avg_buy_price = np.mean(self.prices_of_shares_we_bought) if len(self.prices_of_shares_we_bought) > 0 else 0
            self.total_shares -= 1
            self.prices_of_shares_we_bought = [avg_buy_price for _ in range(self.total_shares)]
            step_reward = (current_price - avg_buy_price)

            self.sell_count += 1
            return step_reward
        
        else:
            """
            The Agent only reachs here if he tries to sell but he has no shares to sell or tries to buy more than the max shares

            If the agent holds for too long, we penalize it by a small amount.
            This is to encourage the agent to not hold for too long, and to
            trade more frequently.
            """
            # if action == Actions.Buy.value:
            #     step_reward = -100
            #     # More logic here to penalize the agent for buying too much
            # elif action == Actions.Sell.value:
            #     step_reward = -100
            #     # More logic here to penalize the agent for selling too much
            self.action_history.append(-99)
            return step_reward

    # ====================================================================
    # Override the TradingEnv functions with our own functions
    # ====================================================================
    def reset(self):
        super().reset() # call TradingEnv.reset()
        self.action_history = []
        self.total_shares = 0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        self._unrealized_gain = 0
        self.prices_of_shares_we_bought = []
        return self._get_observation()

    def step(self, action):
        # Take a step in the environment
        self._done = False
        self._current_tick += 1

        # If we've reached the end of the episode
        if self._current_tick == self._end_tick:
            self._done = True

        # Calculate the reward received in this action
        step_reward = self._calculate_reward(action)

        self._total_reward += step_reward

        # Calculate the total unrealized gain/loss the agent has at the end of the episode
        if self._done:
            current_price = self.prices[self._current_tick]
            avg_buy_price = np.mean(self.prices_of_shares_we_bought) if len(self.prices_of_shares_we_bought) > 0 else 0
            self._unrealized_gain = (current_price - avg_buy_price) * self.total_shares

        # Get the next state
        observation = self._get_observation()
        info = dict(
            step_reward = np.round(step_reward, 2),
            total_reward = np.round(self._total_reward, 2),
            action = action
        )
        self._update_history(info)

        return observation, step_reward, self._done, info
    
    def render_all(self, mode='human'):
        plt.figure(figsize=(15, 6))
        plt.cla()

        window_ticks = np.arange(len(self.action_history))

        plt.plot(self.prices)

        buy_ticks = []
        sell_ticks = []
        hold_ticks = []
        denied_ticks = [] # this is when the agent tries to buy or sell but the environment doesn't allow it to do so
        for i, tick in enumerate(window_ticks):
            if self.action_history[i] == Actions.Buy.value:
                buy_ticks.append(tick)
            elif self.action_history[i] == Actions.Sell.value:
                sell_ticks.append(tick)
            elif self.action_history[i] == Actions.Hold.value:
                hold_ticks.append(tick)
            else:
                denied_ticks.append(tick)
        
        # print(f"len(buy_ticks) = {len(buy_ticks)}, len(sell_ticks) = {len(sell_ticks)}, len(hold_ticks) = {len(hold_ticks)}, len(denied_ticks) = {len(denied_ticks)}")
        plt.plot(buy_ticks, self.prices[buy_ticks], 'g.', label="Buy")
        plt.plot(sell_ticks, self.prices[sell_ticks], 'r.', label="Sell")
        plt.legend()
        # plt.plot(hold_ticks, self.prices[hold_ticks], 'bo', alpha=0.25)
        # plt.plot(denied_ticks, self.prices[denied_ticks], 'bo', alpha=0.25)
        
        plt.title(f"Realized Gain/Loss: {np.round(self._total_reward, 2)}, Unrealized Gain/Loss: {np.round(self._unrealized_gain, 2)}")
        plt.show()