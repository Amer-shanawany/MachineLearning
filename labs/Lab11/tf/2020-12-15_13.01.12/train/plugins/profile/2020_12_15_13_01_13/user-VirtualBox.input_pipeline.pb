	UMu�!@UMu�!@!UMu�!@	뛇aq��?뛇aq��?!뛇aq��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$UMu�!@�$"����?A�.��[� @Y,���o
�?*	H�z�WV@2F
Iterator::Model	��YK�?!��
�RK@)G;n��t�?1�
�z�BE@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatO>=�e��?!��ec�#<@)d"��<�?1)J�Z��4@:Preprocessing2U
Iterator::Model::ParallelMapV2����1�?!^W�)�@(@)����1�?1^W�)�@(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�
�Ov�?!����BD%@)�<,Ԛ�}?1��V�)V @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�o`r��z?!�f�!`D@)�o`r��z?1�f�!`D@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��<���?!j_��/�F@)}�E�j?1MuYES@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��(@̈?!�b�Ω+@)Ih˹We?1śr�Q@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceh��b?!�6�9c�@)h��b?1�6�9c�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9뛇aq��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�$"����?�$"����?!�$"����?      ��!       "      ��!       *      ��!       2	�.��[� @�.��[� @!�.��[� @:      ��!       B      ��!       J	,���o
�?,���o
�?!,���o
�?R      ��!       Z	,���o
�?,���o
�?!,���o
�?JCPU_ONLYY뛇aq��?b 