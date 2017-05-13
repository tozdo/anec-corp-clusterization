import os


def mystem():
    inp_base = '/home/tozdo/anecs/'
    out_base = '/home/tozdo/amystem/'
    for papka_name in os.listdir(path = inp_base):
        for file_name in os.listdir(path = inp_base + papka_name):
                inp = inp_base + papka_name + '/' + file_name
                out = out_base + papka_name + '/' + file_name
                if not os.path.exists(out_base + papka_name):
                    os.makedirs(out_base + papka_name)
                os.system('/home/tozdo/mystem -lnd ' + inp + ' ' + out)

mystem()

