path = "/home/ggc_user/f_count"
def reset():
    try:
        with open(path, 'x'): pass
    except:
        pass

    with open(path, "w+") as f:
        f.write(str(0))

def increment(max):
    exceeded = False
    try:
        with open(path, 'x'): pass
    except:
        pass

    with open(path, "r+") as f:
        try:
            amt = int(f.read())
            f.seek(0)
            if(amt >= max):
                exceeded = True
        except:
            # file didnt exist
            amt = 0
        amt += 1
        f.write(str(amt))
    return exceeded

def get_value():
        try:
            with open(path, "r") as f:
                amt = int(f.read())
                return amt
        except:
            return -1

