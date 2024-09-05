def show_as_minutes(seconds):
    '''
    takes int seconds as input and returns str value
    '''
    min = int(seconds // 60)
    sec = int(seconds - (min * 60))
    return "Total Time: {} min {} sec".format(min, sec)