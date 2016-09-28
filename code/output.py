arr = [[5.45303575e-05,  2.40629444e-03, -5.25896301e-03],
       [-3.65719149e-03,  3.74015137e-03, -1.01803945e-02],
       [-3.88547065e-04, -1.76511100e-03, -9.16523483e-04],
       [-7.07634935e-03, -6.15501473e-03,  5.30956360e-04],
       [-7.63019612e-03,  4.03027023e-04,  6.32087574e-03],
       [-7.63019612e-03, -9.40757936e-04, -8.05271343e-03]]

for result in arr:
    print("{:.02f} {:.02f} {:.02f}".format(
        *[int(elm*100) / 100. for elm in result]))
##
# Output after 40 rounds
#
# 0.00 0.00 0.00
# 0.00 0.00 -0.01
# 0.00 0.00 0.00
# 0.00 0.00 0.00
# 0.00 0.00 0.00
# 0.00 0.00 0.00
