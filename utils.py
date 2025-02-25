def is_pixel_in_center(x_pred, y_pred, x_label, y_label, tolerance):
     return  abs(x_pred-x_label) <= tolerance and abs(y_pred-y_label) <= tolerance

def calPosition(x:int, image_size):
     x_pred = x %  image_size
     y_pred = x // image_size
     return x_pred, y_pred