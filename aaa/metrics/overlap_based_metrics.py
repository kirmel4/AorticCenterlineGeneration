from aaa.metrics.misc import __score_generator

jaccard_score = __score_generator(lambda TP, FP, FN: TP / (TP + FP + FN))
dice_score = __score_generator(lambda TP, FP, FN: 2*TP / (2*TP + FP + FN))
sensitivity_score = __score_generator(lambda TP, FP, FN: TP / (TP + FN))

precision_score = __score_generator(lambda TP, FP, FN: TP / (TP + FP))
recall_score = __score_generator(lambda TP, FP, FN: TP / (TP + FN))

volumetric_similarity_score = __score_generator(lambda TP, FP, FN: 1 - abs(FN - FP) / (2 * TP + FP + FN))
