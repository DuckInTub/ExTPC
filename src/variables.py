# Maybe define global variables here
# I don't know how such things work tough
import math

frame_time = 5.2701 # ms
frame_interval = 200 # ms
total_nr_frames = math.ceil(1000 * 60 / frame_interval)

tx_power_min = -25 # dBm
tx_power_max = 0 # dBm
rx_power_target = -85 # dBm
offset = 3 # dB
avg_weight = 0.8 # alpha in Xiaos paper