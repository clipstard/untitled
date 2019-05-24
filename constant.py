LEFT = 'left'
RIGHT = 'right'
FORWARD = 'forward'
FREE_DRIVE = 'free_drive'
STOP = 'stop'
PARKING = 'parking'
FINISH = 'finish'
BLUE_LIGHT = 'blue_light'
LEFT_YELLOW = 'left_yellow'
RIGHT_YELLOW = 'right_yellow'
STOP_LIGHT = 'stop_light'
NIGHT_LIGHT = 'night_light'
BACK_LIGHT = 'back_light'
IS_DAY = "b\'day_time:0\\n\'"
IS_NIGHT = "b\'day_time:1\\n\'"
OBJECT_IN_FRONT = "b'sens_fata:1\\n'"
OBJECT_IN_BACK = "b'sens_spate:1\\n'"

signals = {
    'blue_light': 19,
    'left_yellow': 6,
    'right_yellow': 5,
    'stop_light': 26,
    'night_light': 13,
    'back_light': 21
}
