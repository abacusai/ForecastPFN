"""
Module containing configuration setting for the script
"""

class Config:
    is_sub_day = False

    @classmethod
    def set_sub_day(cls, is_sub_day):
        Config.is_sub_day = is_sub_day
