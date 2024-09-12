def operation(operation):
    if operation == 1:
        return (False, True)
        # LOAD_PREV_MODEL = False
        # DO_TEST = True
    elif operation == 2:
        return (False, False)
        # LOAD_PREV_MODEL = False
        # DO_TEST = False
    elif operation == 3:
        return (True, True)
        # LOAD_PREV_MODEL = True
        # DO_TEST = True
