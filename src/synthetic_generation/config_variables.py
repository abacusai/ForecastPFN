"""
Module containing configuration setting for the script
"""


class Config:
    frequencies = None
    frequency_names = None
    freq_and_index = None
    transition = False

    @classmethod
    def set_freq_variables(cls, is_sub_day):
        if is_sub_day:
            cls.frequencies = [
                ('min', 1 / 1440),
                ('H', 1 / 24),
                ('D', 1),
                ('W', 7),
                ('MS', 30),
                ('Y', 12),
            ]
            cls.frequency_names = [
                'minute',
                'hourly',
                'daily',
                'weekly',
                'monthly',
                'yearly',
            ]
            cls.freq_and_index = (
                ('minute', 0),
                ('hourly', 1),
                ('daily', 2),
                ('weekly', 3),
                ('monthly', 4),
                ('yearly', 5),
            )
        else:
            cls.frequencies = [('D', 1), ('W', 7), ('MS', 30)]
            cls.frequency_names = ['daily', 'weekly', 'monthly']
            cls.freq_and_index = (('daily', 0), ('weekly', 1), ('monthly', 2))

    @classmethod
    def set_transition(cls, transition):
        cls.transition = transition
