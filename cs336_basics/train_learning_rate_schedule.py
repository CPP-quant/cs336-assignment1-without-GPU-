import math

def learning_rate_schedule(t,max_alpha,min_alpha,T_w,T_c):
    if t<T_w:
        return t/T_w*max_alpha
    elif t<T_c:
        return min_alpha+(1+math.cos(math.pi*(t-T_w)/(T_c-T_w)))*(max_alpha-min_alpha)/2
    else:
        return min_alpha