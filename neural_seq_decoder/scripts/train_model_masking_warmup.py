
modelName = 'BaselineWithMaskingWaarmup_100'

args = {}
args['outputDir'] = '/home/jupyter/competitionData/speech_logs/' + modelName
args['datasetPath'] = '/home/jupyter/competitionData/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 6000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['timeMaskMaxFrac'] = 0.2
args['timeMaskNum'] = 4
args['warmupSteps'] = 100

import sys
sys.path.append('neural_seq_decoder/src/')
from neural_decoder.trainer_time_masking_warmup import trainModel

trainModel(args)