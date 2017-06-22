import evaluation_measures
import numpy as np

purity_nmf_tfidf = []
entropy_nmf_tfidf = []
ari_nmf_tfidf = []
nmf_tfidf = []

purity_nmf_tfidf_cvalue = []
entropy_nmf_tfidf_cvalue = []
ari_nmf_tfidf_cvalue = []
nmf_tfidf_cvalue = []

purity_lda_tfidf = []
entropy_lda_tfidf = []
ari_lda_tfidf = []
lda_tfidf = []

purity_lda_tfidf_cvalue = []
entropy_lda_tfidf_cvalue = []
ari_lda_tfidf_cvalue = []
lda_tfidf_cvalue = []

# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 615, 1: 169, 2: 99, 3: 1655, 4: 2521}, 'medical': {0: 323, 1: 107, 2: 1108, 3: 1473, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 264, 1: 169, 2: 2929, 3: 648, 4: 64}, 'datamining': {0: 1621, 1: 128, 2: 79, 3: 288, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 229, 1: 3350, 2: 1122, 3: 151, 4: 207}, 'medical': {0: 1206, 1: 699, 2: 303, 3: 787, 4: 71}, 'theory': {0: 127, 1: 134, 2: 215, 3: 172, 4: 3347}, 'visu': {0: 171, 1: 289, 2: 401, 3: 3076, 4: 137}, 'datamining': {0: 164, 1: 296, 2: 1576, 3: 126, 4: 108}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2349, 1: 99, 2: 89, 3: 743, 4: 1779}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3322, 2: 137, 3: 207, 4: 200}, 'visu': {0: 50, 1: 82, 2: 2959, 3: 333, 4: 650}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 175, 1: 232, 2: 3584, 3: 130, 4: 938}, 'medical': {0: 70, 1: 694, 2: 107, 3: 872, 4: 1323}, 'theory': {0: 3339, 1: 175, 2: 274, 3: 139, 4: 68}, 'visu': {0: 96, 1: 239, 2: 236, 3: 3095, 4: 408}, 'datamining': {0: 149, 1: 533, 2: 854, 3: 287, 4: 447}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 613, 1: 169, 2: 99, 3: 1665, 4: 2513}, 'medical': {0: 323, 1: 106, 2: 1108, 3: 1474, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 262, 1: 169, 2: 2928, 3: 651, 4: 64}, 'datamining': {0: 1620, 1: 128, 2: 79, 3: 289, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 155, 1: 2583, 2: 1977, 3: 148, 4: 196}, 'medical': {0: 1767, 1: 563, 2: 182, 3: 473, 4: 81}, 'theory': {0: 38, 1: 110, 2: 439, 3: 239, 4: 3169}, 'visu': {0: 405, 1: 379, 2: 305, 3: 2903, 4: 82}, 'datamining': {0: 151, 1: 329, 2: 1470, 3: 226, 4: 94}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2348, 1: 99, 2: 89, 3: 743, 4: 1780}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3321, 2: 137, 3: 207, 4: 201}, 'visu': {0: 50, 1: 82, 2: 2958, 3: 333, 4: 651}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 136, 1: 189, 2: 1937, 3: 240, 4: 2557}, 'medical': {0: 873, 1: 78, 2: 1141, 3: 857, 4: 117}, 'theory': {0: 196, 1: 3070, 2: 85, 3: 118, 4: 526}, 'visu': {0: 2957, 1: 97, 2: 458, 3: 276, 4: 286}, 'datamining': {0: 181, 1: 114, 2: 332, 3: 651, 4: 992}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 614, 1: 169, 2: 99, 3: 1659, 4: 2518}, 'medical': {0: 323, 1: 107, 2: 1108, 3: 1473, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 263, 1: 169, 2: 2929, 3: 649, 4: 64}, 'datamining': {0: 1621, 1: 128, 2: 79, 3: 288, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 3439, 1: 389, 2: 168, 3: 199, 4: 864}, 'medical': {0: 171, 1: 1542, 2: 1031, 3: 70, 4: 252}, 'theory': {0: 228, 1: 54, 2: 163, 3: 3321, 4: 229}, 'visu': {0: 243, 1: 276, 2: 2832, 3: 112, 4: 611}, 'datamining': {0: 277, 1: 237, 2: 143, 3: 102, 4: 1511}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2352, 1: 99, 2: 89, 3: 743, 4: 1776}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3322, 2: 137, 3: 207, 4: 200}, 'visu': {0: 50, 1: 83, 2: 2960, 3: 333, 4: 648}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 1314, 1: 2964, 2: 249, 3: 327, 4: 205}, 'medical': {0: 88, 1: 510, 2: 674, 3: 1687, 4: 107}, 'theory': {0: 899, 1: 78, 2: 185, 3: 125, 4: 2708}, 'visu': {0: 146, 1: 408, 2: 3197, 3: 235, 4: 88}, 'datamining': {0: 425, 1: 856, 2: 541, 3: 338, 4: 110}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 613, 1: 169, 2: 98, 3: 1659, 4: 2520}, 'medical': {0: 323, 1: 107, 2: 1108, 3: 1473, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 261, 1: 169, 2: 2929, 3: 651, 4: 64}, 'datamining': {0: 1620, 1: 128, 2: 79, 3: 289, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 3851, 1: 522, 2: 385, 3: 197, 4: 104}, 'medical': {0: 259, 1: 986, 2: 429, 3: 67, 4: 1325}, 'theory': {0: 263, 1: 92, 2: 203, 3: 3350, 4: 87}, 'visu': {0: 206, 1: 913, 2: 2420, 3: 135, 4: 400}, 'datamining': {0: 940, 1: 168, 2: 735, 3: 143, 4: 284}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2351, 1: 99, 2: 89, 3: 743, 4: 1777}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3322, 2: 137, 3: 207, 4: 200}, 'visu': {0: 50, 1: 83, 2: 2959, 3: 333, 4: 649}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 160, 1: 204, 2: 3818, 3: 205, 4: 672}, 'medical': {0: 504, 1: 49, 2: 186, 3: 849, 4: 1478}, 'theory': {0: 156, 1: 3448, 2: 159, 3: 185, 4: 47}, 'visu': {0: 130, 1: 82, 2: 327, 3: 3182, 4: 353}, 'datamining': {0: 254, 1: 119, 2: 1217, 3: 404, 4: 276}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 614, 1: 169, 2: 99, 3: 1658, 4: 2519}, 'medical': {0: 323, 1: 107, 2: 1108, 3: 1473, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 264, 1: 169, 2: 2929, 3: 648, 4: 64}, 'datamining': {0: 1621, 1: 128, 2: 79, 3: 288, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 2371, 1: 2158, 2: 198, 3: 143, 4: 189}, 'medical': {0: 135, 1: 1238, 2: 429, 3: 79, 4: 1185}, 'theory': {0: 606, 1: 91, 2: 194, 3: 2982, 4: 122}, 'visu': {0: 217, 1: 374, 2: 3132, 3: 97, 4: 254}, 'datamining': {0: 804, 1: 645, 2: 445, 3: 130, 4: 246}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2350, 1: 99, 2: 89, 3: 743, 4: 1778}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3322, 2: 137, 3: 207, 4: 200}, 'visu': {0: 50, 1: 82, 2: 2959, 3: 333, 4: 650}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 257, 1: 173, 2: 3912, 3: 532, 4: 185}, 'medical': {0: 61, 1: 530, 2: 908, 3: 681, 4: 886}, 'theory': {0: 3446, 1: 136, 2: 160, 3: 172, 4: 81}, 'visu': {0: 101, 1: 2608, 2: 485, 3: 287, 4: 593}, 'datamining': {0: 109, 1: 217, 2: 485, 3: 1340, 4: 119}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 615, 1: 169, 2: 99, 3: 1662, 4: 2514}, 'medical': {0: 323, 1: 107, 2: 1108, 3: 1473, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 264, 1: 169, 2: 2929, 3: 648, 4: 64}, 'datamining': {0: 1621, 1: 128, 2: 79, 3: 288, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 178, 1: 2234, 2: 217, 3: 148, 4: 2282}, 'medical': {0: 153, 1: 182, 2: 59, 3: 1349, 4: 1323}, 'theory': {0: 1480, 1: 178, 2: 2082, 3: 153, 4: 102}, 'visu': {0: 123, 1: 524, 2: 199, 3: 2717, 4: 511}, 'datamining': {0: 162, 1: 273, 2: 181, 3: 388, 4: 1266}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2349, 1: 99, 2: 89, 3: 743, 4: 1779}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3321, 2: 137, 3: 207, 4: 201}, 'visu': {0: 50, 1: 82, 2: 2959, 3: 333, 4: 650}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 331, 1: 146, 2: 472, 3: 3985, 4: 125}, 'medical': {0: 63, 1: 1200, 2: 1198, 3: 361, 4: 244}, 'theory': {0: 3346, 1: 108, 2: 176, 3: 196, 4: 169}, 'visu': {0: 122, 1: 2058, 2: 148, 3: 362, 4: 1384}, 'datamining': {0: 106, 1: 320, 2: 278, 3: 1413, 4: 153}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 613, 1: 169, 2: 99, 3: 1665, 4: 2513}, 'medical': {0: 322, 1: 106, 2: 1108, 3: 1475, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 261, 1: 169, 2: 2928, 3: 652, 4: 64}, 'datamining': {0: 1620, 1: 128, 2: 79, 3: 289, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 1678, 1: 123, 2: 2907, 3: 251, 4: 100}, 'medical': {0: 145, 1: 1268, 2: 348, 3: 1225, 4: 80}, 'theory': {0: 677, 1: 173, 2: 121, 3: 83, 4: 2941}, 'visu': {0: 398, 1: 2642, 2: 330, 3: 554, 4: 150}, 'datamining': {0: 848, 1: 188, 2: 650, 3: 474, 4: 110}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2348, 1: 98, 2: 89, 3: 743, 4: 1781}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3321, 2: 137, 3: 207, 4: 201}, 'visu': {0: 50, 1: 82, 2: 2958, 3: 333, 4: 651}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 112, 1: 197, 2: 3533, 3: 624, 4: 593}, 'medical': {0: 54, 1: 837, 2: 203, 3: 1411, 4: 561}, 'theory': {0: 3395, 1: 99, 2: 225, 3: 95, 4: 181}, 'visu': {0: 131, 1: 1165, 2: 191, 3: 222, 4: 2365}, 'datamining': {0: 107, 1: 123, 2: 329, 3: 378, 4: 1333}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 614, 1: 169, 2: 99, 3: 1658, 4: 2519}, 'medical': {0: 323, 1: 107, 2: 1108, 3: 1473, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 264, 1: 169, 2: 2929, 3: 648, 4: 64}, 'datamining': {0: 1621, 1: 128, 2: 79, 3: 288, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 3706, 1: 378, 2: 196, 3: 634, 4: 145}, 'medical': {0: 183, 1: 527, 2: 1061, 3: 1233, 4: 62}, 'theory': {0: 231, 1: 131, 2: 313, 3: 65, 4: 3255}, 'visu': {0: 317, 1: 156, 2: 2799, 3: 693, 4: 109}, 'datamining': {0: 1302, 1: 302, 2: 327, 3: 199, 4: 140}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2351, 1: 99, 2: 89, 3: 743, 4: 1777}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3322, 2: 137, 3: 207, 4: 200}, 'visu': {0: 50, 1: 82, 2: 2959, 3: 333, 4: 650}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 243, 1: 4068, 2: 219, 3: 334, 4: 195}, 'medical': {0: 636, 1: 643, 2: 262, 3: 1455, 4: 70}, 'theory': {0: 169, 1: 191, 2: 119, 3: 128, 4: 3388}, 'visu': {0: 3147, 1: 444, 2: 194, 3: 193, 4: 96}, 'datamining': {0: 630, 1: 1142, 2: 51, 3: 264, 4: 183}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 615, 1: 169, 2: 99, 3: 1664, 4: 2512}, 'medical': {0: 324, 1: 107, 2: 1108, 3: 1473, 4: 54}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 264, 1: 169, 2: 2928, 3: 649, 4: 64}, 'datamining': {0: 1621, 1: 128, 2: 79, 3: 288, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 111, 1: 187, 2: 351, 3: 1436, 4: 2974}, 'medical': {0: 124, 1: 1243, 2: 417, 3: 1162, 4: 120}, 'theory': {0: 3151, 1: 143, 2: 122, 3: 121, 4: 458}, 'visu': {0: 158, 1: 2188, 2: 1219, 3: 303, 4: 206}, 'datamining': {0: 122, 1: 230, 2: 527, 3: 445, 4: 946}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2351, 1: 99, 2: 89, 3: 743, 4: 1777}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3322, 2: 137, 3: 207, 4: 200}, 'visu': {0: 50, 1: 82, 2: 2959, 3: 333, 4: 650}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 187, 1: 3385, 2: 296, 3: 138, 4: 1053}, 'medical': {0: 615, 1: 367, 2: 1709, 3: 64, 4: 311}, 'theory': {0: 202, 1: 158, 2: 97, 3: 3314, 4: 224}, 'visu': {0: 2924, 1: 398, 2: 254, 3: 68, 4: 430}, 'datamining': {0: 148, 1: 296, 2: 168, 3: 82, 4: 1576}})
# NMF TFIDF with cvalue:
nmf_tfidf_cvalue.append({'database': {0: 614, 1: 169, 2: 99, 3: 1661, 4: 2516}, 'medical': {0: 323, 1: 107, 2: 1108, 3: 1473, 4: 55}, 'theory': {0: 111, 1: 3440, 2: 128, 3: 211, 4: 105}, 'visu': {0: 263, 1: 169, 2: 2929, 3: 649, 4: 64}, 'datamining': {0: 1621, 1: 128, 2: 79, 3: 288, 4: 154}})
# LDA TFIDF with cvalue:
lda_tfidf_cvalue.append({'database': {0: 135, 1: 501, 2: 163, 3: 927, 4: 3333}, 'medical': {0: 70, 1: 1362, 2: 986, 3: 503, 4: 145}, 'theory': {0: 3325, 1: 82, 2: 204, 3: 132, 4: 252}, 'visu': {0: 125, 1: 223, 2: 2995, 3: 528, 4: 203}, 'datamining': {0: 151, 1: 187, 2: 188, 3: 1496, 4: 248}})
# NMF TFIDF without cvalue:
nmf_tfidf.append({'database': {0: 2351, 1: 99, 2: 89, 3: 743, 4: 1777}, 'medical': {0: 34, 1: 78, 2: 1152, 3: 373, 4: 1429}, 'theory': {0: 129, 1: 3322, 2: 137, 3: 207, 4: 200}, 'visu': {0: 50, 1: 82, 2: 2959, 3: 333, 4: 650}, 'datamining': {0: 91, 1: 56, 2: 42, 3: 1831, 4: 250}})
# LDA TFIDF without cvalue:
lda_tfidf.append({'database': {0: 3007, 1: 205, 2: 137, 3: 158, 4: 1552}, 'medical': {0: 552, 1: 81, 2: 906, 3: 1062, 4: 465}, 'theory': {0: 127, 1: 3416, 2: 150, 3: 106, 4: 196}, 'visu': {0: 314, 1: 139, 2: 1298, 3: 1934, 4: 389}, 'datamining': {0: 228, 1: 96, 2: 114, 3: 151, 4: 1681}})



purity_nmf_okapi = []
entropy_nmf_okapi = []
ari_nmf_okapi = []
nmf_okapi = []

purity_nmf_okapi_cvalue = []
entropy_nmf_okapi_cvalue = []
ari_nmf_okapi_cvalue = []
nmf_okapi_cvalue = []

purity_lda_okapi = []
entropy_lda_okapi = []
ari_lda_okapi = []
lda_okapi = []

purity_lda_okapi_cvalue = []
entropy_lda_okapi_cvalue = []
ari_lda_okapi_cvalue = []
lda_okapi_cvalue = []


# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 641, 1: 1046, 2: 93, 3: 636, 4: 2643}, 'medical': {0: 117, 1: 253, 2: 1210, 3: 1287, 4: 199}, 'theory': {0: 3618, 1: 121, 2: 75, 3: 102, 4: 79}, 'visu': {0: 434, 1: 754, 2: 2529, 3: 211, 4: 146}, 'datamining': {0: 398, 1: 1466, 2: 124, 3: 113, 4: 169}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 170, 1: 3357, 2: 1170, 3: 132, 4: 230}, 'medical': {0: 567, 1: 322, 2: 225, 3: 158, 4: 1794}, 'theory': {0: 162, 1: 196, 2: 377, 3: 3196, 4: 64}, 'visu': {0: 2784, 1: 374, 2: 475, 3: 164, 4: 277}, 'datamining': {0: 151, 1: 499, 2: 1331, 3: 131, 4: 158}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3556, 1: 677, 2: 89, 3: 61, 4: 676}, 'medical': {0: 213, 1: 161, 2: 1275, 3: 15, 4: 1402}, 'theory': {0: 90, 1: 3442, 2: 63, 3: 291, 4: 109}, 'visu': {0: 578, 1: 728, 2: 2539, 3: 15, 4: 214}, 'datamining': {0: 1276, 1: 721, 2: 132, 3: 25, 4: 116}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 383, 1: 1973, 2: 2357, 3: 150, 4: 196}, 'medical': {0: 1210, 1: 261, 2: 331, 3: 1201, 4: 63}, 'theory': {0: 61, 1: 279, 2: 237, 3: 125, 4: 3293}, 'visu': {0: 279, 1: 208, 2: 1848, 3: 1550, 4: 189}, 'datamining': {0: 243, 1: 206, 2: 1581, 3: 122, 4: 118}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 623, 1: 1036, 2: 91, 3: 625, 4: 2684}, 'medical': {0: 116, 1: 253, 2: 1211, 3: 1287, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 430, 1: 756, 2: 2529, 3: 209, 4: 150}, 'datamining': {0: 397, 1: 1467, 2: 123, 3: 112, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 82, 1: 3554, 2: 498, 3: 652, 4: 273}, 'medical': {0: 1019, 1: 213, 2: 1269, 3: 454, 4: 111}, 'theory': {0: 147, 1: 175, 2: 77, 3: 216, 4: 3380}, 'visu': {0: 1668, 1: 285, 2: 216, 3: 1747, 4: 158}, 'datamining': {0: 90, 1: 376, 2: 480, 3: 1237, 4: 87}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3555, 1: 677, 2: 89, 3: 61, 4: 677}, 'medical': {0: 213, 1: 161, 2: 1274, 3: 15, 4: 1403}, 'theory': {0: 89, 1: 3441, 2: 63, 3: 291, 4: 111}, 'visu': {0: 577, 1: 729, 2: 2538, 3: 15, 4: 215}, 'datamining': {0: 1274, 1: 722, 2: 132, 3: 25, 4: 117}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 3744, 1: 320, 2: 159, 3: 189, 4: 647}, 'medical': {0: 217, 1: 794, 2: 113, 3: 478, 4: 1464}, 'theory': {0: 351, 1: 239, 2: 3155, 3: 195, 4: 55}, 'visu': {0: 455, 1: 242, 2: 141, 3: 2988, 4: 248}, 'datamining': {0: 1105, 1: 117, 2: 136, 3: 423, 4: 489}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 625, 1: 1037, 2: 91, 3: 626, 4: 2680}, 'medical': {0: 116, 1: 253, 2: 1211, 3: 1287, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 430, 1: 756, 2: 2529, 3: 209, 4: 150}, 'datamining': {0: 397, 1: 1467, 2: 123, 3: 112, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 280, 1: 221, 2: 2681, 3: 98, 4: 1779}, 'medical': {0: 1340, 1: 146, 2: 429, 3: 894, 4: 257}, 'theory': {0: 85, 1: 2950, 2: 92, 3: 226, 4: 642}, 'visu': {0: 205, 1: 177, 2: 525, 3: 2339, 4: 828}, 'datamining': {0: 259, 1: 108, 2: 444, 3: 116, 4: 1343}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3555, 1: 677, 2: 89, 3: 61, 4: 677}, 'medical': {0: 213, 1: 161, 2: 1275, 3: 15, 4: 1402}, 'theory': {0: 89, 1: 3441, 2: 63, 3: 291, 4: 111}, 'visu': {0: 577, 1: 729, 2: 2538, 3: 15, 4: 215}, 'datamining': {0: 1275, 1: 722, 2: 132, 3: 25, 4: 116}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 1405, 1: 135, 2: 2209, 3: 788, 4: 522}, 'medical': {0: 75, 1: 1182, 2: 1035, 3: 600, 4: 174}, 'theory': {0: 3228, 1: 183, 2: 61, 3: 94, 4: 429}, 'visu': {0: 487, 1: 2156, 2: 317, 3: 975, 4: 139}, 'datamining': {0: 322, 1: 161, 2: 248, 3: 1439, 4: 100}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 624, 1: 1039, 2: 91, 3: 626, 4: 2679}, 'medical': {0: 116, 1: 253, 2: 1212, 3: 1286, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 430, 1: 758, 2: 2527, 3: 209, 4: 150}, 'datamining': {0: 397, 1: 1467, 2: 123, 3: 112, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 3409, 1: 558, 2: 250, 3: 629, 4: 213}, 'medical': {0: 439, 1: 981, 2: 147, 3: 154, 4: 1345}, 'theory': {0: 131, 1: 104, 2: 2591, 3: 1056, 4: 113}, 'visu': {0: 926, 1: 143, 2: 739, 3: 270, 4: 1996}, 'datamining': {0: 1451, 1: 171, 2: 262, 3: 165, 4: 221}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3555, 1: 677, 2: 89, 3: 61, 4: 677}, 'medical': {0: 213, 1: 161, 2: 1275, 3: 15, 4: 1402}, 'theory': {0: 89, 1: 3441, 2: 63, 3: 291, 4: 111}, 'visu': {0: 577, 1: 729, 2: 2538, 3: 15, 4: 215}, 'datamining': {0: 1275, 1: 721, 2: 132, 3: 25, 4: 117}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 387, 1: 196, 2: 1694, 3: 2536, 4: 246}, 'medical': {0: 955, 1: 1363, 2: 170, 3: 499, 4: 79}, 'theory': {0: 93, 1: 115, 2: 271, 3: 125, 4: 3391}, 'visu': {0: 194, 1: 2280, 2: 205, 3: 1167, 4: 228}, 'datamining': {0: 201, 1: 215, 2: 130, 3: 1567, 4: 157}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 622, 1: 1038, 2: 91, 3: 626, 4: 2682}, 'medical': {0: 116, 1: 253, 2: 1211, 3: 1287, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 430, 1: 759, 2: 2526, 3: 209, 4: 150}, 'datamining': {0: 397, 1: 1468, 2: 122, 3: 112, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 3640, 1: 585, 2: 192, 3: 183, 4: 459}, 'medical': {0: 296, 1: 560, 2: 92, 3: 1178, 4: 940}, 'theory': {0: 211, 1: 284, 2: 3235, 3: 159, 4: 106}, 'visu': {0: 463, 1: 266, 2: 150, 3: 3018, 4: 177}, 'datamining': {0: 440, 1: 1391, 2: 145, 3: 165, 4: 129}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3556, 1: 677, 2: 89, 3: 61, 4: 676}, 'medical': {0: 214, 1: 161, 2: 1275, 3: 15, 4: 1401}, 'theory': {0: 90, 1: 3443, 2: 63, 3: 291, 4: 108}, 'visu': {0: 578, 1: 729, 2: 2538, 3: 15, 4: 214}, 'datamining': {0: 1276, 1: 721, 2: 132, 3: 25, 4: 116}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 1421, 1: 2013, 2: 1319, 3: 152, 4: 154}, 'medical': {0: 1136, 1: 278, 2: 412, 3: 79, 4: 1161}, 'theory': {0: 54, 1: 459, 2: 191, 3: 3215, 4: 76}, 'visu': {0: 228, 1: 283, 2: 2697, 3: 293, 4: 573}, 'datamining': {0: 468, 1: 205, 2: 1326, 3: 106, 4: 165}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 635, 1: 1039, 2: 92, 3: 625, 4: 2668}, 'medical': {0: 117, 1: 252, 2: 1214, 3: 1284, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 433, 1: 752, 2: 2533, 3: 207, 4: 149}, 'datamining': {0: 401, 1: 1465, 2: 124, 3: 110, 4: 170}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 253, 1: 242, 2: 4054, 3: 304, 4: 206}, 'medical': {0: 775, 1: 1055, 2: 509, 3: 650, 4: 77}, 'theory': {0: 102, 1: 95, 2: 212, 3: 132, 4: 3454}, 'visu': {0: 381, 1: 2353, 2: 453, 3: 612, 4: 275}, 'datamining': {0: 447, 1: 227, 2: 1285, 3: 135, 4: 176}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3555, 1: 677, 2: 89, 3: 61, 4: 677}, 'medical': {0: 213, 1: 161, 2: 1275, 3: 15, 4: 1402}, 'theory': {0: 89, 1: 3441, 2: 63, 3: 291, 4: 111}, 'visu': {0: 577, 1: 729, 2: 2538, 3: 15, 4: 215}, 'datamining': {0: 1275, 1: 721, 2: 132, 3: 25, 4: 117}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 3711, 1: 244, 2: 406, 3: 180, 4: 518}, 'medical': {0: 259, 1: 995, 2: 232, 3: 87, 4: 1493}, 'theory': {0: 462, 1: 114, 2: 127, 3: 3250, 4: 42}, 'visu': {0: 1083, 1: 2144, 2: 238, 3: 226, 4: 383}, 'datamining': {0: 1464, 1: 405, 2: 122, 3: 94, 4: 185}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 624, 1: 1040, 2: 91, 3: 626, 4: 2678}, 'medical': {0: 116, 1: 253, 2: 1211, 3: 1287, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 430, 1: 758, 2: 2527, 3: 209, 4: 150}, 'datamining': {0: 397, 1: 1467, 2: 123, 3: 112, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 3200, 1: 351, 2: 613, 3: 198, 4: 697}, 'medical': {0: 478, 1: 726, 2: 75, 3: 1015, 4: 772}, 'theory': {0: 121, 1: 153, 2: 3502, 3: 155, 4: 64}, 'visu': {0: 528, 1: 235, 2: 183, 3: 2938, 4: 190}, 'datamining': {0: 614, 1: 1044, 2: 271, 3: 218, 4: 123}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3556, 1: 677, 2: 89, 3: 61, 4: 676}, 'medical': {0: 213, 1: 161, 2: 1275, 3: 15, 4: 1402}, 'theory': {0: 90, 1: 3441, 2: 63, 3: 291, 4: 110}, 'visu': {0: 578, 1: 729, 2: 2538, 3: 15, 4: 214}, 'datamining': {0: 1276, 1: 721, 2: 132, 3: 25, 4: 116}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 261, 1: 113, 2: 2437, 3: 2074, 4: 174}, 'medical': {0: 180, 1: 1459, 2: 269, 3: 1096, 4: 62}, 'theory': {0: 157, 1: 93, 2: 257, 3: 62, 4: 3426}, 'visu': {0: 181, 1: 1923, 2: 1579, 3: 255, 4: 136}, 'datamining': {0: 109, 1: 175, 2: 1464, 3: 403, 4: 119}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 628, 1: 1041, 2: 91, 3: 627, 4: 2672}, 'medical': {0: 117, 1: 253, 2: 1210, 3: 1287, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 431, 1: 759, 2: 2526, 3: 209, 4: 149}, 'datamining': {0: 397, 1: 1468, 2: 122, 3: 112, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 928, 1: 2869, 2: 104, 3: 852, 4: 306}, 'medical': {0: 429, 1: 206, 2: 1257, 3: 1103, 4: 71}, 'theory': {0: 221, 1: 146, 2: 101, 3: 115, 4: 3412}, 'visu': {0: 1485, 1: 149, 2: 1983, 3: 285, 4: 172}, 'datamining': {0: 1252, 1: 471, 2: 123, 3: 288, 4: 136}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3552, 1: 677, 2: 89, 3: 61, 4: 680}, 'medical': {0: 212, 1: 161, 2: 1274, 3: 15, 4: 1404}, 'theory': {0: 89, 1: 3441, 2: 63, 3: 291, 4: 111}, 'visu': {0: 577, 1: 730, 2: 2537, 3: 15, 4: 215}, 'datamining': {0: 1274, 1: 722, 2: 132, 3: 25, 4: 117}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 137, 1: 747, 2: 148, 3: 651, 4: 3376}, 'medical': {0: 1210, 1: 361, 2: 74, 3: 1118, 4: 303}, 'theory': {0: 98, 1: 201, 2: 3389, 3: 101, 4: 206}, 'visu': {0: 2762, 1: 280, 2: 156, 3: 203, 4: 673}, 'datamining': {0: 90, 1: 1346, 2: 95, 3: 166, 4: 573}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 618, 1: 1041, 2: 92, 3: 625, 4: 2683}, 'medical': {0: 115, 1: 253, 2: 1212, 3: 1287, 4: 199}, 'theory': {0: 3614, 1: 122, 2: 75, 3: 102, 4: 82}, 'visu': {0: 427, 1: 762, 2: 2525, 3: 209, 4: 151}, 'datamining': {0: 395, 1: 1469, 2: 122, 3: 113, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 2414, 1: 181, 2: 236, 3: 1913, 4: 315}, 'medical': {0: 90, 1: 1015, 2: 245, 3: 1137, 4: 579}, 'theory': {0: 1374, 1: 224, 2: 2166, 3: 69, 4: 162}, 'visu': {0: 213, 1: 3003, 2: 165, 3: 461, 4: 232}, 'datamining': {0: 388, 1: 274, 2: 266, 3: 480, 4: 862}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3555, 1: 677, 2: 89, 3: 61, 4: 677}, 'medical': {0: 213, 1: 161, 2: 1275, 3: 15, 4: 1402}, 'theory': {0: 89, 1: 3441, 2: 63, 3: 291, 4: 111}, 'visu': {0: 577, 1: 729, 2: 2538, 3: 15, 4: 215}, 'datamining': {0: 1275, 1: 721, 2: 132, 3: 25, 4: 117}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 925, 1: 156, 2: 159, 3: 3536, 4: 283}, 'medical': {0: 550, 1: 69, 2: 782, 3: 248, 4: 1417}, 'theory': {0: 55, 1: 3360, 2: 179, 3: 345, 4: 56}, 'visu': {0: 264, 1: 153, 2: 2844, 3: 538, 4: 275}, 'datamining': {0: 417, 1: 153, 2: 237, 3: 1142, 4: 321}})
# NMF Okapi with cvalue:
nmf_okapi_cvalue.append({'database': {0: 624, 1: 1039, 2: 91, 3: 627, 4: 2678}, 'medical': {0: 116, 1: 253, 2: 1211, 3: 1287, 4: 199}, 'theory': {0: 3615, 1: 121, 2: 75, 3: 102, 4: 82}, 'visu': {0: 430, 1: 759, 2: 2526, 3: 209, 4: 150}, 'datamining': {0: 396, 1: 1468, 2: 122, 3: 113, 4: 171}})
# LDA Okapi with cvalue:
lda_okapi_cvalue.append({'database': {0: 233, 1: 130, 2: 1753, 3: 506, 4: 2437}, 'medical': {0: 73, 1: 1378, 2: 161, 3: 979, 4: 475}, 'theory': {0: 3286, 1: 89, 2: 346, 3: 128, 4: 146}, 'visu': {0: 371, 1: 2183, 2: 124, 3: 170, 4: 1226}, 'datamining': {0: 173, 1: 194, 2: 177, 3: 457, 4: 1269}})
# NMF Okapi without cvalue:
nmf_okapi.append({'database': {0: 3556, 1: 677, 2: 89, 3: 61, 4: 676}, 'medical': {0: 213, 1: 161, 2: 1275, 3: 15, 4: 1402}, 'theory': {0: 90, 1: 3442, 2: 63, 3: 291, 4: 109}, 'visu': {0: 578, 1: 729, 2: 2538, 3: 15, 4: 214}, 'datamining': {0: 1276, 1: 721, 2: 132, 3: 25, 4: 116}})
# LDA Okapi without cvalue:
lda_okapi.append({'database': {0: 1142, 1: 153, 2: 3096, 3: 150, 4: 518}, 'medical': {0: 302, 1: 1242, 2: 320, 3: 80, 4: 1122}, 'theory': {0: 101, 1: 113, 2: 313, 3: 3377, 4: 91}, 'visu': {0: 252, 1: 2671, 2: 800, 3: 166, 4: 185}, 'datamining': {0: 215, 1: 200, 2: 1566, 3: 114, 4: 175}})


for elem in nmf_tfidf:
    entropy_nmf_tfidf.append(evaluation_measures.entropy(elem))
    purity_nmf_tfidf.append(evaluation_measures.purity(elem))
    ari_nmf_tfidf.append(evaluation_measures.adj_rand_index(elem))

for elem in nmf_tfidf_cvalue:
    entropy_nmf_tfidf_cvalue.append(evaluation_measures.entropy(elem))
    purity_nmf_tfidf_cvalue.append(evaluation_measures.purity(elem))
    ari_nmf_tfidf_cvalue.append(evaluation_measures.adj_rand_index(elem))

for elem in lda_tfidf:
    entropy_lda_tfidf.append(evaluation_measures.entropy(elem))
    purity_lda_tfidf.append(evaluation_measures.purity(elem))
    ari_lda_tfidf.append(evaluation_measures.adj_rand_index(elem))

for elem in lda_tfidf_cvalue:
    entropy_lda_tfidf_cvalue.append(evaluation_measures.entropy(elem))
    purity_lda_tfidf_cvalue.append(evaluation_measures.purity(elem))    
    ari_lda_tfidf_cvalue.append(evaluation_measures.adj_rand_index(elem))



for elem in nmf_okapi:
    entropy_nmf_okapi.append(evaluation_measures.entropy(elem))
    purity_nmf_okapi.append(evaluation_measures.purity(elem))
    ari_nmf_okapi.append(evaluation_measures.adj_rand_index(elem))

for elem in nmf_okapi_cvalue:
    entropy_nmf_okapi_cvalue.append(evaluation_measures.entropy(elem))
    purity_nmf_okapi_cvalue.append(evaluation_measures.purity(elem))
    ari_nmf_okapi_cvalue.append(evaluation_measures.adj_rand_index(elem))

for elem in lda_okapi:
    entropy_lda_okapi.append(evaluation_measures.entropy(elem))
    purity_lda_okapi.append(evaluation_measures.purity(elem))
    ari_lda_okapi.append(evaluation_measures.adj_rand_index(elem))

for elem in lda_okapi_cvalue:
    entropy_lda_okapi_cvalue.append(evaluation_measures.entropy(elem))
    purity_lda_okapi_cvalue.append(evaluation_measures.purity(elem))
    ari_lda_okapi_cvalue.append(evaluation_measures.adj_rand_index(elem)) 


print("ART LDA TFIDF Entropy without c-value:", round(np.mean(entropy_lda_tfidf), 2), "+/-", round(np.std(entropy_lda_tfidf), 2))
print("ART LDA Okapi Entropy without c-value:", round(np.mean(entropy_lda_okapi), 2), "+/-", round(np.std(entropy_lda_okapi), 2))
print("ART NMF TFIDF Entropy without c-value:", round(np.mean(entropy_nmf_tfidf), 2), "+/-", round(np.std(entropy_nmf_tfidf), 2))
print("ART NMF Okapi Entropy without c-value:", round(np.mean(entropy_nmf_okapi), 2), "+/-", round(np.std(entropy_nmf_okapi), 2))

print("ART LDA TFIDF Entropy with    c-value:", round(np.mean(entropy_lda_tfidf_cvalue), 2), "+/-", round(np.std(entropy_lda_tfidf_cvalue), 2))
print("ART LDA Okapi Entropy with    c-value:", round(np.mean(entropy_lda_okapi_cvalue), 2), "+/-", round(np.std(entropy_lda_okapi_cvalue), 2))
print("ART NMF TFIDF Entropy with    c-value:", round(np.mean(entropy_nmf_tfidf_cvalue), 2), "+/-", round(np.std(entropy_nmf_tfidf_cvalue), 2))
print("ART NMF Okapi Entropy with    c-value:", round(np.mean(entropy_nmf_okapi_cvalue), 2), "+/-", round(np.std(entropy_nmf_okapi_cvalue), 2))


print("ART LDA TFIDF Purity  without c-value:", round(np.mean(purity_lda_tfidf), 2), "+/-", round(np.std(purity_lda_tfidf), 2))
print("ART LDA Okapi Purity  without c-value:", round(np.mean(purity_lda_okapi), 2), "+/-", round(np.std(purity_lda_okapi), 2))
print("ART NMF TFIDF Purity  without c-value:", round(np.mean(purity_nmf_tfidf), 2), "+/-", round(np.std(purity_nmf_tfidf), 2))
print("ART NMF Okapi Purity  without c-value:", round(np.mean(purity_nmf_okapi), 2), "+/-", round(np.std(purity_nmf_okapi), 2))

print("ART LDA TFIDF Purity  with    c-value:", round(np.mean(purity_lda_tfidf_cvalue), 2), "+/-", round(np.std(purity_lda_tfidf_cvalue), 2))
print("ART LDA Okapi Purity  with    c-value:", round(np.mean(purity_lda_okapi_cvalue), 2), "+/-", round(np.std(purity_lda_okapi_cvalue), 2))
print("ART NMF TFIDF Purity  with    c-value:", round(np.mean(purity_nmf_tfidf_cvalue), 2), "+/-", round(np.std(purity_nmf_tfidf_cvalue), 2))
print("ART NMF Okapi Purity  with    c-value:", round(np.mean(purity_nmf_okapi_cvalue), 2), "+/-", round(np.std(purity_nmf_okapi_cvalue), 2))

print("ART LDA TFIDF ARI     without c-value:", round(np.mean(ari_lda_tfidf), 2), "+/-", round(np.std(ari_lda_tfidf), 2))
print("ART LDA Okapi ARI     without c-value:", round(np.mean(ari_lda_okapi), 2), "+/-", round(np.std(ari_lda_okapi), 2))
print("ART NMF TFIDF ARI     without c-value:", round(np.mean(ari_nmf_tfidf), 2), "+/-", round(np.std(ari_nmf_tfidf), 2))
print("ART NMF Okapi ARI     without c-value:", round(np.mean(ari_nmf_okapi), 2), "+/-", round(np.std(ari_nmf_okapi), 2))

print("ART LDA TFIDF ARI     with    c-value:", round(np.mean(ari_lda_tfidf_cvalue), 2), "+/-", round(np.std(ari_lda_tfidf_cvalue), 2))
print("ART LDA Okapi ARI     with    c-value:", round(np.mean(ari_lda_okapi_cvalue), 2), "+/-", round(np.std(ari_lda_okapi_cvalue), 2))
print("ART NMF TFIDF ARI     with    c-value:", round(np.mean(ari_nmf_tfidf_cvalue), 2), "+/-", round(np.std(ari_nmf_tfidf_cvalue), 2))
print("ART NMF Okapi ARI     with    c-value:", round(np.mean(ari_nmf_okapi_cvalue), 2), "+/-", round(np.std(ari_nmf_okapi_cvalue), 2))

