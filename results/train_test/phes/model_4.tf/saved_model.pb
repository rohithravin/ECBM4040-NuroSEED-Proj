û
Ġ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
+
IsInf
x"T
y
"
Ttype:
2
,
Log
x"T
y"T"
Ttype:

2


LogicalNot
x

y

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint˙˙˙˙˙˙˙˙˙"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
À
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8?
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
ô8*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
ô8*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
ô8*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
Ë 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 
valueüBù Bò
˘
siamese_network
loss_tracker
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
4
	total
	count
	variables
	keras_api
d
iter

beta_1

beta_2
	decay
learning_ratemambvcvd
 

0
1
 

0
1
2
3
­
trainable_variables
metrics
layer_regularization_losses
regularization_losses

layers
	variables
 non_trainable_variables
!layer_metrics
 
 
 
 
"layer-0
#layer-1
$layer-2
%layer_with_weights-0
%layer-3
&trainable_variables
'regularization_losses
(	variables
)	keras_api
R
*trainable_variables
+regularization_losses
,	variables
-	keras_api

0
1
 

0
1
­
trainable_variables
.metrics
/layer_regularization_losses
regularization_losses

0layers
	variables
1non_trainable_variables
2layer_metrics
HF
VARIABLE_VALUEtotal-loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEcount-loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_4/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE

0
 

0

0
1


loss
R
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api

0
1
 

0
1
­
&trainable_variables
Cmetrics
Dlayer_regularization_losses
'regularization_losses

Elayers
(	variables
Fnon_trainable_variables
Glayer_metrics
 
 
 
­
*trainable_variables
Hmetrics
Ilayer_regularization_losses

Jlayers
+regularization_losses
,	variables
Knon_trainable_variables
Llayer_metrics
 
 


0
1
2
3
 
 
 
 
 
­
3trainable_variables
Mmetrics
Nlayer_regularization_losses

Olayers
4regularization_losses
5	variables
Pnon_trainable_variables
Qlayer_metrics
 
 
 
­
7trainable_variables
Rmetrics
Slayer_regularization_losses

Tlayers
8regularization_losses
9	variables
Unon_trainable_variables
Vlayer_metrics
 
 
 
­
;trainable_variables
Wmetrics
Xlayer_regularization_losses

Ylayers
<regularization_losses
=	variables
Znon_trainable_variables
[layer_metrics

0
1
 

0
1
­
?trainable_variables
\metrics
]layer_regularization_losses

^layers
@regularization_losses
A	variables
_non_trainable_variables
`layer_metrics
 
 

"0
#1
$2
%3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEAdam/dense_4/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_4/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_4/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_4/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
|
serving_default_input_2Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
ï
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2dense_4/kerneldense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_20978719
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_20979543
ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcount	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_4/kerneldense_4/biasAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_20979592ŜÑ
Ĥ
?
cond_false_20979254
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ċ

*__inference_dense_4_layer_call_fn_20979480

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_209783322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙ô8::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8
 
_user_specified_nameinputs
	
Ŝ
E__inference_dense_4_layer_call_and_return_conditional_losses_20979471

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙ô8::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8
 
_user_specified_nameinputs
?
Ĉ
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978682	
input
input_1
model_4_20978676
model_4_20978678
identity˘model_4/StatefulPartitionedCall
model_4/StatefulPartitionedCallStatefulPartitionedCallinputinput_1model_4_20978676model_4_20978678*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_209786312!
model_4/StatefulPartitionedCall
IdentityIdentity(model_4/StatefulPartitionedCall:output:0 ^model_4/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
½
m
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_20979427
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
?

*__inference_model_4_layer_call_fn_20979161
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_209785582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
·

&__inference_signature_wrapper_20978719
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCallŬ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_209782572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
?

*__inference_model_4_layer_call_fn_20979079
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_209786312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1

0
__inference_call_20978173
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
˘.
×
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978781
input_1
input_2?
;model_4_sequential_4_dense_4_matmul_readvariableop_resource@
<model_4_sequential_4_dense_4_biasadd_readvariableop_resource
identity˘3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp˘5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘2model_4/sequential_4/dense_4/MatMul/ReadVariableOp˘4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp
'model_4/sequential_4/dropout_4/IdentityIdentityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_4/sequential_4/dropout_4/Identity?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall0model_4/sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall
$model_4/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_4/sequential_4/flatten_4/Const
&model_4/sequential_4/flatten_4/ReshapeReshapeFmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0-model_4/sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_4/sequential_4/flatten_4/Reshapeĉ
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpô
#model_4/sequential_4/dense_4/MatMulMatMul/model_4/sequential_4/flatten_4/Reshape:output:0:model_4/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_4/sequential_4/dense_4/MatMulä
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpö
$model_4/sequential_4/dense_4/BiasAddBiasAdd-model_4/sequential_4/dense_4/MatMul:product:0;model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_4/sequential_4/dense_4/BiasAdd
)model_4/sequential_4/dropout_4/Identity_1Identityinput_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_4/sequential_4/dropout_4/Identity_1Ğ
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall2model_4/sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732A
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1Ħ
&model_4/sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_4/sequential_4/flatten_4/Const_1
(model_4/sequential_4/flatten_4/Reshape_1ReshapeHmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0/model_4/sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_4/sequential_4/flatten_4/Reshape_1ê
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpü
%model_4/sequential_4/dense_4/MatMul_1MatMul1model_4/sequential_4/flatten_4/Reshape_1:output:0<model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_4/sequential_4/dense_4/MatMul_1è
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpŝ
&model_4/sequential_4/dense_4/BiasAdd_1BiasAdd/model_4/sequential_4/dense_4/MatMul_1:product:0=model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_4/sequential_4/dense_4/BiasAdd_1Ħ
(model_4/distance_layer_4/PartitionedCallPartitionedCall-model_4/sequential_4/dense_4/BiasAdd:output:0/model_4/sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472*
(model_4/distance_layer_4/PartitionedCallÛ
IdentityIdentity1model_4/distance_layer_4/PartitionedCall:output:04^model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp6^model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp3^model_4/sequential_4/dense_4/MatMul/ReadVariableOp5^model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp2n
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp2h
2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2l
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
öG
×
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978839
input_0
input_1?
;model_4_sequential_4_dense_4_matmul_readvariableop_resource@
<model_4_sequential_4_dense_4_biasadd_readvariableop_resource
identity˘3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp˘5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘2model_4/sequential_4/dense_4/MatMul/ReadVariableOp˘4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpĦ
,model_4/sequential_4/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_4/sequential_4/dropout_4/dropout/ConstÒ
*model_4/sequential_4/dropout_4/dropout/MulMulinput_05model_4/sequential_4/dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_4/sequential_4/dropout_4/dropout/Mul
,model_4/sequential_4/dropout_4/dropout/ShapeShapeinput_0*
T0*
_output_shapes
:2.
,model_4/sequential_4/dropout_4/dropout/Shape
Cmodel_4/sequential_4/dropout_4/dropout/random_uniform/RandomUniformRandomUniform5model_4/sequential_4/dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_4/sequential_4/dropout_4/dropout/random_uniform/RandomUniform³
5model_4/sequential_4/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_4/sequential_4/dropout_4/dropout/GreaterEqual/yğ
3model_4/sequential_4/dropout_4/dropout/GreaterEqualGreaterEqualLmodel_4/sequential_4/dropout_4/dropout/random_uniform/RandomUniform:output:0>model_4/sequential_4/dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_4/sequential_4/dropout_4/dropout/GreaterEqualŬ
+model_4/sequential_4/dropout_4/dropout/CastCast7model_4/sequential_4/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_4/sequential_4/dropout_4/dropout/Cast÷
,model_4/sequential_4/dropout_4/dropout/Mul_1Mul.model_4/sequential_4/dropout_4/dropout/Mul:z:0/model_4/sequential_4/dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_4/sequential_4/dropout_4/dropout/Mul_1?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall0model_4/sequential_4/dropout_4/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall
$model_4/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_4/sequential_4/flatten_4/Const
&model_4/sequential_4/flatten_4/ReshapeReshapeFmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0-model_4/sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_4/sequential_4/flatten_4/Reshapeĉ
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpô
#model_4/sequential_4/dense_4/MatMulMatMul/model_4/sequential_4/flatten_4/Reshape:output:0:model_4/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_4/sequential_4/dense_4/MatMulä
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpö
$model_4/sequential_4/dense_4/BiasAddBiasAdd-model_4/sequential_4/dense_4/MatMul:product:0;model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_4/sequential_4/dense_4/BiasAdd?
.model_4/sequential_4/dropout_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_4/sequential_4/dropout_4/dropout_1/ConstĜ
,model_4/sequential_4/dropout_4/dropout_1/MulMulinput_17model_4/sequential_4/dropout_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_4/sequential_4/dropout_4/dropout_1/Mul
.model_4/sequential_4/dropout_4/dropout_1/ShapeShapeinput_1*
T0*
_output_shapes
:20
.model_4/sequential_4/dropout_4/dropout_1/Shape
Emodel_4/sequential_4/dropout_4/dropout_1/random_uniform/RandomUniformRandomUniform7model_4/sequential_4/dropout_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_4/sequential_4/dropout_4/dropout_1/random_uniform/RandomUniform·
7model_4/sequential_4/dropout_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_4/sequential_4/dropout_4/dropout_1/GreaterEqual/y?
5model_4/sequential_4/dropout_4/dropout_1/GreaterEqualGreaterEqualNmodel_4/sequential_4/dropout_4/dropout_1/random_uniform/RandomUniform:output:0@model_4/sequential_4/dropout_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_4/sequential_4/dropout_4/dropout_1/GreaterEqual?
-model_4/sequential_4/dropout_4/dropout_1/CastCast9model_4/sequential_4/dropout_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_4/sequential_4/dropout_4/dropout_1/Cast˙
.model_4/sequential_4/dropout_4/dropout_1/Mul_1Mul0model_4/sequential_4/dropout_4/dropout_1/Mul:z:01model_4/sequential_4/dropout_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_4/sequential_4/dropout_4/dropout_1/Mul_1Ğ
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall2model_4/sequential_4/dropout_4/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732A
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1Ħ
&model_4/sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_4/sequential_4/flatten_4/Const_1
(model_4/sequential_4/flatten_4/Reshape_1ReshapeHmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0/model_4/sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_4/sequential_4/flatten_4/Reshape_1ê
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpü
%model_4/sequential_4/dense_4/MatMul_1MatMul1model_4/sequential_4/flatten_4/Reshape_1:output:0<model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_4/sequential_4/dense_4/MatMul_1è
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpŝ
&model_4/sequential_4/dense_4/BiasAdd_1BiasAdd/model_4/sequential_4/dense_4/MatMul_1:product:0=model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_4/sequential_4/dense_4/BiasAdd_1Ħ
(model_4/distance_layer_4/PartitionedCallPartitionedCall-model_4/sequential_4/dense_4/BiasAdd:output:0/model_4/sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472*
(model_4/distance_layer_4/PartitionedCallÛ
IdentityIdentity1model_4/distance_layer_4/PartitionedCall:output:04^model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp6^model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp3^model_4/sequential_4/dense_4/MatMul/ReadVariableOp5^model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp2n
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp2h
2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2l
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
Ĥ
?
cond_false_20979374
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ğ
e
,__inference_dropout_4_layer_call_fn_20979413

inputs
identity˘StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_209782732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ŝ
E__inference_dense_4_layer_call_and_return_conditional_losses_20978332

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙ô8::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8
 
_user_specified_nameinputs

·
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978397

inputs
dense_4_20978391
dense_4_20978393
identity˘dense_4/StatefulPartitionedCallŬ
dropout_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_209782782
dropout_4/PartitionedCallŞ
(one_hot_encoding_layer_4/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_209783002*
(one_hot_encoding_layer_4/PartitionedCall
flatten_4/PartitionedCallPartitionedCall1one_hot_encoding_layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_209783142
flatten_4/PartitionedCallµ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_20978391dense_4_20978393*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_209783322!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

9
cond_true_20978229
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

9
cond_true_20979253
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

9
cond_true_20979373
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?)

E__inference_model_4_layer_call_and_return_conditional_losses_20978631

inputs
inputs_17
3sequential_4_dense_4_matmul_readvariableop_resource8
4sequential_4_dense_4_biasadd_readvariableop_resource
identity˘+sequential_4/dense_4/BiasAdd/ReadVariableOp˘-sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘*sequential_4/dense_4/MatMul/ReadVariableOp˘,sequential_4/dense_4/MatMul_1/ReadVariableOp
sequential_4/dropout_4/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_4/dropout_4/Identity
5sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall(sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817327
5sequential_4/one_hot_encoding_layer_4/PartitionedCall
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_4/flatten_4/Constċ
sequential_4/flatten_4/ReshapeReshape>sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_4/flatten_4/ReshapeÎ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOpÔ
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMulÌ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOpÖ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/BiasAdd
!sequential_4/dropout_4/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_4/dropout_4/Identity_1
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall*sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817329
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1
sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_4/flatten_4/Const_1í
 sequential_4/flatten_4/Reshape_1Reshape@sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0'sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_4/flatten_4/Reshape_1Ò
,sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_4/dense_4/MatMul_1/ReadVariableOpÜ
sequential_4/dense_4/MatMul_1MatMul)sequential_4/flatten_4/Reshape_1:output:04sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMul_1?
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpŜ
sequential_4/dense_4/BiasAdd_1BiasAdd'sequential_4/dense_4/MatMul_1:product:05sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_4/dense_4/BiasAdd_1
 distance_layer_4/PartitionedCallPartitionedCall%sequential_4/dense_4/BiasAdd:output:0'sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472"
 distance_layer_4/PartitionedCall³
IdentityIdentity)distance_layer_4/PartitionedCall:output:0,^sequential_4/dense_4/BiasAdd/ReadVariableOp.^sequential_4/dense_4/BiasAdd_1/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp-^sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2^
-sequential_4/dense_4/BiasAdd_1/ReadVariableOp-sequential_4/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2\
,sequential_4/dense_4/MatMul_1/ReadVariableOp,sequential_4/dense_4/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
í

2__inference_siamese_model_4_layer_call_fn_20978873
input_0
input_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_209786822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1

?
cond_false_20979317
cond_identity_add
cond_identityc
cond/IdentityIdentitycond_identity_add*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:
Â
Û
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978376

inputs
dense_4_20978370
dense_4_20978372
identity˘dense_4/StatefulPartitionedCall˘!dropout_4/StatefulPartitionedCallġ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_209782732#
!dropout_4/StatefulPartitionedCall²
(one_hot_encoding_layer_4/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_209783002*
(one_hot_encoding_layer_4/PartitionedCall
flatten_4/PartitionedCallPartitionedCall1one_hot_encoding_layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_209783142
flatten_4/PartitionedCallµ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_20978370dense_4_20978372*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_209783322!
dense_4/StatefulPartitionedCall?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ó

/__inference_sequential_4_layer_call_fn_20978383
input_5
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
é

*__inference_model_4_layer_call_fn_20978565
	sequence1
	sequence2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	sequence1	sequence2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_209785582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2
ĥ
ü
E__inference_model_4_layer_call_and_return_conditional_losses_20978502
	sequence1
	sequence2
sequential_4_20978427
sequential_4_20978429
identity˘$sequential_4/StatefulPartitionedCall˘&sequential_4/StatefulPartitionedCall_1µ
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_4_20978427sequential_4_20978429*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783762&
$sequential_4/StatefulPartitionedCallı
&sequential_4/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_4_20978427sequential_4_20978429*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783762(
&sequential_4/StatefulPartitionedCall_1Ĉ
 distance_layer_4/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0/sequential_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_209784922"
 distance_layer_4/PartitionedCallÉ
IdentityIdentity)distance_layer_4/PartitionedCall:output:0%^sequential_4/StatefulPartitionedCall'^sequential_4/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2P
&sequential_4/StatefulPartitionedCall_1&sequential_4/StatefulPartitionedCall_1:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2
ĝ

J__inference_sequential_4_layer_call_and_return_conditional_losses_20979196

inputs*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity˘dense_4/BiasAdd/ReadVariableOp˘dense_4/MatMul/ReadVariableOpo
dropout_4/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_4/Identityĉ
(one_hot_encoding_layer_4/PartitionedCallPartitionedCalldropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732*
(one_hot_encoding_layer_4/PartitionedCalls
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_4/Constħ
flatten_4/ReshapeReshape1one_hot_encoding_layer_4/PartitionedCall:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_4/Reshape§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_4/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp˘
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_4/BiasAdd?
IdentityIdentitydense_4/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨
ĝ
E__inference_model_4_layer_call_and_return_conditional_losses_20978558

inputs
inputs_1
sequential_4_20978548
sequential_4_20978550
identity˘$sequential_4/StatefulPartitionedCall˘&sequential_4/StatefulPartitionedCall_1²
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_20978548sequential_4_20978550*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783972&
$sequential_4/StatefulPartitionedCall¸
&sequential_4/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_4_20978548sequential_4_20978550*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783972(
&sequential_4/StatefulPartitionedCall_1Ĉ
 distance_layer_4/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0/sequential_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_209784922"
 distance_layer_4/PartitionedCallÉ
IdentityIdentity)distance_layer_4/PartitionedCall:output:0%^sequential_4/StatefulPartitionedCall'^sequential_4/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2P
&sequential_4/StatefulPartitionedCall_1&sequential_4/StatefulPartitionedCall_1:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĝ)
Ħ
E__inference_model_4_layer_call_and_return_conditional_losses_20979141
inputs_0
inputs_17
3sequential_4_dense_4_matmul_readvariableop_resource8
4sequential_4_dense_4_biasadd_readvariableop_resource
identity˘+sequential_4/dense_4/BiasAdd/ReadVariableOp˘-sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘*sequential_4/dense_4/MatMul/ReadVariableOp˘,sequential_4/dense_4/MatMul_1/ReadVariableOp
sequential_4/dropout_4/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_4/dropout_4/Identity
5sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall(sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817327
5sequential_4/one_hot_encoding_layer_4/PartitionedCall
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_4/flatten_4/Constċ
sequential_4/flatten_4/ReshapeReshape>sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_4/flatten_4/ReshapeÎ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOpÔ
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMulÌ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOpÖ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/BiasAdd
!sequential_4/dropout_4/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_4/dropout_4/Identity_1
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall*sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817329
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1
sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_4/flatten_4/Const_1í
 sequential_4/flatten_4/Reshape_1Reshape@sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0'sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_4/flatten_4/Reshape_1Ò
,sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_4/dense_4/MatMul_1/ReadVariableOpÜ
sequential_4/dense_4/MatMul_1MatMul)sequential_4/flatten_4/Reshape_1:output:04sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMul_1?
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpŜ
sequential_4/dense_4/BiasAdd_1BiasAdd'sequential_4/dense_4/MatMul_1:product:05sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_4/dense_4/BiasAdd_1
 distance_layer_4/PartitionedCallPartitionedCall%sequential_4/dense_4/BiasAdd:output:0'sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472"
 distance_layer_4/PartitionedCall³
IdentityIdentity)distance_layer_4/PartitionedCall:output:0,^sequential_4/dense_4/BiasAdd/ReadVariableOp.^sequential_4/dense_4/BiasAdd_1/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp-^sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2^
-sequential_4/dense_4/BiasAdd_1/ReadVariableOp-sequential_4/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2\
,sequential_4/dense_4/MatMul_1/ReadVariableOp,sequential_4/dense_4/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
˘.
×
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978863
input_0
input_1?
;model_4_sequential_4_dense_4_matmul_readvariableop_resource@
<model_4_sequential_4_dense_4_biasadd_readvariableop_resource
identity˘3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp˘5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘2model_4/sequential_4/dense_4/MatMul/ReadVariableOp˘4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp
'model_4/sequential_4/dropout_4/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_4/sequential_4/dropout_4/Identity?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall0model_4/sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall
$model_4/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_4/sequential_4/flatten_4/Const
&model_4/sequential_4/flatten_4/ReshapeReshapeFmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0-model_4/sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_4/sequential_4/flatten_4/Reshapeĉ
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpô
#model_4/sequential_4/dense_4/MatMulMatMul/model_4/sequential_4/flatten_4/Reshape:output:0:model_4/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_4/sequential_4/dense_4/MatMulä
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpö
$model_4/sequential_4/dense_4/BiasAddBiasAdd-model_4/sequential_4/dense_4/MatMul:product:0;model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_4/sequential_4/dense_4/BiasAdd
)model_4/sequential_4/dropout_4/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_4/sequential_4/dropout_4/Identity_1Ğ
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall2model_4/sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732A
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1Ħ
&model_4/sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_4/sequential_4/flatten_4/Const_1
(model_4/sequential_4/flatten_4/Reshape_1ReshapeHmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0/model_4/sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_4/sequential_4/flatten_4/Reshape_1ê
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpü
%model_4/sequential_4/dense_4/MatMul_1MatMul1model_4/sequential_4/flatten_4/Reshape_1:output:0<model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_4/sequential_4/dense_4/MatMul_1è
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpŝ
&model_4/sequential_4/dense_4/BiasAdd_1BiasAdd/model_4/sequential_4/dense_4/MatMul_1:product:0=model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_4/sequential_4/dense_4/BiasAdd_1Ħ
(model_4/distance_layer_4/PartitionedCallPartitionedCall-model_4/sequential_4/dense_4/BiasAdd:output:0/model_4/sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472*
(model_4/distance_layer_4/PartitionedCallÛ
IdentityIdentity1model_4/distance_layer_4/PartitionedCall:output:04^model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp6^model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp3^model_4/sequential_4/dense_4/MatMul/ReadVariableOp5^model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp2n
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp2h
2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2l
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
Ô"
n
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_20978492
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNotı
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *&
else_branchR
cond_false_20978475*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_209784742
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
ı
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_20978314

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82

Identity"
identityIdentity:output:0*+
_input_shapes
:˙˙˙˙˙˙˙˙˙:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

H
,__inference_dropout_4_layer_call_fn_20979418

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_209782782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĥ
R
;__inference_one_hot_encoding_layer_4_layer_call_fn_20979432
x
identity×
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_209783002
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
?

/__inference_sequential_4_layer_call_fn_20979205

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĥ
ü
E__inference_model_4_layer_call_and_return_conditional_losses_20978516
	sequence1
	sequence2
sequential_4_20978506
sequential_4_20978508
identity˘$sequential_4/StatefulPartitionedCall˘&sequential_4/StatefulPartitionedCall_1µ
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_4_20978506sequential_4_20978508*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783972&
$sequential_4/StatefulPartitionedCallı
&sequential_4/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_4_20978506sequential_4_20978508*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783972(
&sequential_4/StatefulPartitionedCall_1Ĉ
 distance_layer_4/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0/sequential_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_209784922"
 distance_layer_4/PartitionedCallÉ
IdentityIdentity)distance_layer_4/PartitionedCall:output:0%^sequential_4/StatefulPartitionedCall'^sequential_4/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2P
&sequential_4/StatefulPartitionedCall_1&sequential_4/StatefulPartitionedCall_1:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2

?
cond_false_20978953
cond_identity_add
cond_identityc
cond/IdentityIdentitycond_identity_add*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:
?

*__inference_model_4_layer_call_fn_20979069
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_209786072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
í

2__inference_siamese_model_4_layer_call_fn_20978791
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_209786822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
é

*__inference_model_4_layer_call_fn_20978541
	sequence1
	sequence2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	sequence1	sequence2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_209785342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2
í

2__inference_siamese_model_4_layer_call_fn_20978883
input_0
input_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_209786822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
ż 
9
__inference_call_20979334
s1
s2
identityD
subSubs1s2*
T0* 
_output_shapes
:
2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yU
powPowsub:z:0pow/y:output:0*
T0* 
_output_shapes
:
2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indices`
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yV
pow_1Pows2pow_1/y:output:0*
T0* 
_output_shapes
:
2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesh
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yV
pow_2Pows1pow_2/y:output:0*
T0* 
_output_shapes
:
2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesh
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/x]
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*
_output_shapes	
:2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Const^
MaximumMaximum	sub_1:z:0Const:output:0*
T0*
_output_shapes	
:2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/x]
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*
_output_shapes	
:2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1d
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*
_output_shapes	
:2
	Maximum_1S
mulMulMaximum:z:0Maximum_1:z:0*
T0*
_output_shapes	
:2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2b
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*
_output_shapes	
:2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x[
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*
_output_shapes	
:2
mul_1]
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*
_output_shapes	
:2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/xV
addAddV2add/x:output:0truediv:z:0*
T0*
_output_shapes	
:2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/yV
pow_3Powadd:z:0pow_3/y:output:0*
T0*
_output_shapes	
:2
pow_3H
IsInfIsInf	pow_3:z:0*
T0*
_output_shapes	
:2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNotİ
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes	
:* 
_read_only_resource_inputs
 *&
else_branchR
cond_false_20979317*
output_shapes	
:*%
then_branchR
cond_true_209793162
cond_
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes	
:2
cond/Identity^
add_1AddV2add:z:0cond/Identity:output:0*
T0*
_output_shapes	
:2
add_1B
LogLog	add_1:z:0*
T0*
_output_shapes	
:2
LogO
IdentityIdentityLog:y:0*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*+
_input_shapes
:
:
:D @
 
_output_shapes
:


_user_specified_names1:D@
 
_output_shapes
:


_user_specified_names2
Ĝ)
Ħ
E__inference_model_4_layer_call_and_return_conditional_losses_20979059
inputs_0
inputs_17
3sequential_4_dense_4_matmul_readvariableop_resource8
4sequential_4_dense_4_biasadd_readvariableop_resource
identity˘+sequential_4/dense_4/BiasAdd/ReadVariableOp˘-sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘*sequential_4/dense_4/MatMul/ReadVariableOp˘,sequential_4/dense_4/MatMul_1/ReadVariableOp
sequential_4/dropout_4/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_4/dropout_4/Identity
5sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall(sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817327
5sequential_4/one_hot_encoding_layer_4/PartitionedCall
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_4/flatten_4/Constċ
sequential_4/flatten_4/ReshapeReshape>sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_4/flatten_4/ReshapeÎ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOpÔ
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMulÌ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOpÖ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/BiasAdd
!sequential_4/dropout_4/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_4/dropout_4/Identity_1
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall*sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817329
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1
sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_4/flatten_4/Const_1í
 sequential_4/flatten_4/Reshape_1Reshape@sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0'sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_4/flatten_4/Reshape_1Ò
,sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_4/dense_4/MatMul_1/ReadVariableOpÜ
sequential_4/dense_4/MatMul_1MatMul)sequential_4/flatten_4/Reshape_1:output:04sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMul_1?
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpŜ
sequential_4/dense_4/BiasAdd_1BiasAdd'sequential_4/dense_4/MatMul_1:product:05sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_4/dense_4/BiasAdd_1
 distance_layer_4/PartitionedCallPartitionedCall%sequential_4/dense_4/BiasAdd:output:0'sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472"
 distance_layer_4/PartitionedCall³
IdentityIdentity)distance_layer_4/PartitionedCall:output:0,^sequential_4/dense_4/BiasAdd/ReadVariableOp.^sequential_4/dense_4/BiasAdd_1/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp-^sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2^
-sequential_4/dense_4/BiasAdd_1/ReadVariableOp-sequential_4/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2\
,sequential_4/dense_4/MatMul_1/ReadVariableOp,sequential_4/dense_4/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1


J__inference_sequential_4_layer_call_and_return_conditional_losses_20979182

inputs*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity˘dense_4/BiasAdd/ReadVariableOp˘dense_4/MatMul/ReadVariableOpw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_4/dropout/Const
dropout_4/dropout/MulMulinputs dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_4/dropout/Mulh
dropout_4/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_4/dropout/ShapeÓ
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_4/dropout/GreaterEqual/yç
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
dropout_4/dropout/GreaterEqual
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_4/dropout/Cast£
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_4/dropout/Mul_1ĉ
(one_hot_encoding_layer_4/PartitionedCallPartitionedCalldropout_4/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732*
(one_hot_encoding_layer_4/PartitionedCalls
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_4/Constħ
flatten_4/ReshapeReshape1one_hot_encoding_layer_4/PartitionedCall:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_4/Reshape§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_4/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp˘
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_4/BiasAdd?
IdentityIdentitydense_4/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
?

/__inference_sequential_4_layer_call_fn_20979214

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨
ĝ
E__inference_model_4_layer_call_and_return_conditional_losses_20978534

inputs
inputs_1
sequential_4_20978524
sequential_4_20978526
identity˘$sequential_4/StatefulPartitionedCall˘&sequential_4/StatefulPartitionedCall_1²
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_20978524sequential_4_20978526*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783762&
$sequential_4/StatefulPartitionedCall¸
&sequential_4/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_4_20978524sequential_4_20978526*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783762(
&sequential_4/StatefulPartitionedCall_1Ĉ
 distance_layer_4/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0/sequential_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_209784922"
 distance_layer_4/PartitionedCallÉ
IdentityIdentity)distance_layer_4/PartitionedCall:output:0%^sequential_4/StatefulPartitionedCall'^sequential_4/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2P
&sequential_4/StatefulPartitionedCall_1&sequential_4/StatefulPartitionedCall_1:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ı
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_20979456

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82

Identity"
identityIdentity:output:0*+
_input_shapes
:˙˙˙˙˙˙˙˙˙:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
â
9
cond_true_20978952
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yd
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*
_output_shapes	
:2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yd
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*
_output_shapes	
:2

cond/subR
	cond/SqrtSqrtcond/sub:z:0*
T0*
_output_shapes	
:2
	cond/Sqrt_
cond/IdentityIdentitycond/Sqrt:y:0*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:
Î
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_20979408

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
"
9
__inference_call_20979391
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNotı
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *&
else_branchR
cond_false_20979374*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_209793732
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
Ŝ@
Ħ
E__inference_model_4_layer_call_and_return_conditional_losses_20979117
inputs_0
inputs_17
3sequential_4_dense_4_matmul_readvariableop_resource8
4sequential_4_dense_4_biasadd_readvariableop_resource
identity˘+sequential_4/dense_4/BiasAdd/ReadVariableOp˘-sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘*sequential_4/dense_4/MatMul/ReadVariableOp˘,sequential_4/dense_4/MatMul_1/ReadVariableOp
$sequential_4/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_4/dropout_4/dropout/Constğ
"sequential_4/dropout_4/dropout/MulMulinputs_0-sequential_4/dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_4/dropout_4/dropout/Mul
$sequential_4/dropout_4/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_4/dropout_4/dropout/Shapeú
;sequential_4/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_4/dropout_4/dropout/random_uniform/RandomUniform£
-sequential_4/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_4/dropout_4/dropout/GreaterEqual/y
+sequential_4/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_4/dropout_4/dropout/GreaterEqualĊ
#sequential_4/dropout_4/dropout/CastCast/sequential_4/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_4/dropout_4/dropout/Cast×
$sequential_4/dropout_4/dropout/Mul_1Mul&sequential_4/dropout_4/dropout/Mul:z:0'sequential_4/dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_4/dropout_4/dropout/Mul_1
5sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall(sequential_4/dropout_4/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817327
5sequential_4/one_hot_encoding_layer_4/PartitionedCall
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_4/flatten_4/Constċ
sequential_4/flatten_4/ReshapeReshape>sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_4/flatten_4/ReshapeÎ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOpÔ
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMulÌ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOpÖ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/BiasAdd
&sequential_4/dropout_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_4/dropout_4/dropout_1/ConstÁ
$sequential_4/dropout_4/dropout_1/MulMulinputs_1/sequential_4/dropout_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_4/dropout_4/dropout_1/Mul
&sequential_4/dropout_4/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_4/dropout_4/dropout_1/Shape
=sequential_4/dropout_4/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_4/dropout_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_4/dropout_4/dropout_1/random_uniform/RandomUniform§
/sequential_4/dropout_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_4/dropout_4/dropout_1/GreaterEqual/y£
-sequential_4/dropout_4/dropout_1/GreaterEqualGreaterEqualFsequential_4/dropout_4/dropout_1/random_uniform/RandomUniform:output:08sequential_4/dropout_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_4/dropout_4/dropout_1/GreaterEqualË
%sequential_4/dropout_4/dropout_1/CastCast1sequential_4/dropout_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_4/dropout_4/dropout_1/Castß
&sequential_4/dropout_4/dropout_1/Mul_1Mul(sequential_4/dropout_4/dropout_1/Mul:z:0)sequential_4/dropout_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_4/dropout_4/dropout_1/Mul_1
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall*sequential_4/dropout_4/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817329
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1
sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_4/flatten_4/Const_1í
 sequential_4/flatten_4/Reshape_1Reshape@sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0'sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_4/flatten_4/Reshape_1Ò
,sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_4/dense_4/MatMul_1/ReadVariableOpÜ
sequential_4/dense_4/MatMul_1MatMul)sequential_4/flatten_4/Reshape_1:output:04sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMul_1?
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpŜ
sequential_4/dense_4/BiasAdd_1BiasAdd'sequential_4/dense_4/MatMul_1:product:05sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_4/dense_4/BiasAdd_1
 distance_layer_4/PartitionedCallPartitionedCall%sequential_4/dense_4/BiasAdd:output:0'sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472"
 distance_layer_4/PartitionedCall³
IdentityIdentity)distance_layer_4/PartitionedCall:output:0,^sequential_4/dense_4/BiasAdd/ReadVariableOp.^sequential_4/dense_4/BiasAdd_1/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp-^sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2^
-sequential_4/dense_4/BiasAdd_1/ReadVariableOp-sequential_4/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2\
,sequential_4/dense_4/MatMul_1/ReadVariableOp,sequential_4/dense_4/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
Ĝ
0
__inference_call_20979450
x
identityQ
CastCastx*

DstT0*

SrcT0* 
_output_shapes
:
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*$
_output_shapes
:2	
one_hota
IdentityIdentityone_hot:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
:
:C ?
 
_output_shapes
:


_user_specified_namex

9
cond_true_20978474
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_20978278

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĥ
?
cond_false_20978475
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĉ-
Ħ
__inference_call_20978250	
input
input_1?
;model_4_sequential_4_dense_4_matmul_readvariableop_resource@
<model_4_sequential_4_dense_4_biasadd_readvariableop_resource
identity˘3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp˘5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘2model_4/sequential_4/dense_4/MatMul/ReadVariableOp˘4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp
'model_4/sequential_4/dropout_4/IdentityIdentityinput*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_4/sequential_4/dropout_4/Identity?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall0model_4/sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall
$model_4/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_4/sequential_4/flatten_4/Const
&model_4/sequential_4/flatten_4/ReshapeReshapeFmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0-model_4/sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_4/sequential_4/flatten_4/Reshapeĉ
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpô
#model_4/sequential_4/dense_4/MatMulMatMul/model_4/sequential_4/flatten_4/Reshape:output:0:model_4/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_4/sequential_4/dense_4/MatMulä
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpö
$model_4/sequential_4/dense_4/BiasAddBiasAdd-model_4/sequential_4/dense_4/MatMul:product:0;model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_4/sequential_4/dense_4/BiasAdd
)model_4/sequential_4/dropout_4/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_4/sequential_4/dropout_4/Identity_1Ğ
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall2model_4/sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732A
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1Ħ
&model_4/sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_4/sequential_4/flatten_4/Const_1
(model_4/sequential_4/flatten_4/Reshape_1ReshapeHmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0/model_4/sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_4/sequential_4/flatten_4/Reshape_1ê
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpü
%model_4/sequential_4/dense_4/MatMul_1MatMul1model_4/sequential_4/flatten_4/Reshape_1:output:0<model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_4/sequential_4/dense_4/MatMul_1è
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpŝ
&model_4/sequential_4/dense_4/BiasAdd_1BiasAdd/model_4/sequential_4/dense_4/MatMul_1:product:0=model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_4/sequential_4/dense_4/BiasAdd_1Ħ
(model_4/distance_layer_4/PartitionedCallPartitionedCall-model_4/sequential_4/dense_4/BiasAdd:output:0/model_4/sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472*
(model_4/distance_layer_4/PartitionedCallÛ
IdentityIdentity1model_4/distance_layer_4/PartitionedCall:output:04^model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp6^model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp3^model_4/sequential_4/dense_4/MatMul/ReadVariableOp5^model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp2n
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp2h
2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2l
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
öG
×
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978757
input_1
input_2?
;model_4_sequential_4_dense_4_matmul_readvariableop_resource@
<model_4_sequential_4_dense_4_biasadd_readvariableop_resource
identity˘3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp˘5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘2model_4/sequential_4/dense_4/MatMul/ReadVariableOp˘4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpĦ
,model_4/sequential_4/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_4/sequential_4/dropout_4/dropout/ConstÒ
*model_4/sequential_4/dropout_4/dropout/MulMulinput_15model_4/sequential_4/dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_4/sequential_4/dropout_4/dropout/Mul
,model_4/sequential_4/dropout_4/dropout/ShapeShapeinput_1*
T0*
_output_shapes
:2.
,model_4/sequential_4/dropout_4/dropout/Shape
Cmodel_4/sequential_4/dropout_4/dropout/random_uniform/RandomUniformRandomUniform5model_4/sequential_4/dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_4/sequential_4/dropout_4/dropout/random_uniform/RandomUniform³
5model_4/sequential_4/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_4/sequential_4/dropout_4/dropout/GreaterEqual/yğ
3model_4/sequential_4/dropout_4/dropout/GreaterEqualGreaterEqualLmodel_4/sequential_4/dropout_4/dropout/random_uniform/RandomUniform:output:0>model_4/sequential_4/dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_4/sequential_4/dropout_4/dropout/GreaterEqualŬ
+model_4/sequential_4/dropout_4/dropout/CastCast7model_4/sequential_4/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_4/sequential_4/dropout_4/dropout/Cast÷
,model_4/sequential_4/dropout_4/dropout/Mul_1Mul.model_4/sequential_4/dropout_4/dropout/Mul:z:0/model_4/sequential_4/dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_4/sequential_4/dropout_4/dropout/Mul_1?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall0model_4/sequential_4/dropout_4/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall
$model_4/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_4/sequential_4/flatten_4/Const
&model_4/sequential_4/flatten_4/ReshapeReshapeFmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0-model_4/sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_4/sequential_4/flatten_4/Reshapeĉ
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpô
#model_4/sequential_4/dense_4/MatMulMatMul/model_4/sequential_4/flatten_4/Reshape:output:0:model_4/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_4/sequential_4/dense_4/MatMulä
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpö
$model_4/sequential_4/dense_4/BiasAddBiasAdd-model_4/sequential_4/dense_4/MatMul:product:0;model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_4/sequential_4/dense_4/BiasAdd?
.model_4/sequential_4/dropout_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_4/sequential_4/dropout_4/dropout_1/ConstĜ
,model_4/sequential_4/dropout_4/dropout_1/MulMulinput_27model_4/sequential_4/dropout_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_4/sequential_4/dropout_4/dropout_1/Mul
.model_4/sequential_4/dropout_4/dropout_1/ShapeShapeinput_2*
T0*
_output_shapes
:20
.model_4/sequential_4/dropout_4/dropout_1/Shape
Emodel_4/sequential_4/dropout_4/dropout_1/random_uniform/RandomUniformRandomUniform7model_4/sequential_4/dropout_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_4/sequential_4/dropout_4/dropout_1/random_uniform/RandomUniform·
7model_4/sequential_4/dropout_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_4/sequential_4/dropout_4/dropout_1/GreaterEqual/y?
5model_4/sequential_4/dropout_4/dropout_1/GreaterEqualGreaterEqualNmodel_4/sequential_4/dropout_4/dropout_1/random_uniform/RandomUniform:output:0@model_4/sequential_4/dropout_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_4/sequential_4/dropout_4/dropout_1/GreaterEqual?
-model_4/sequential_4/dropout_4/dropout_1/CastCast9model_4/sequential_4/dropout_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_4/sequential_4/dropout_4/dropout_1/Cast˙
.model_4/sequential_4/dropout_4/dropout_1/Mul_1Mul0model_4/sequential_4/dropout_4/dropout_1/Mul:z:01model_4/sequential_4/dropout_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_4/sequential_4/dropout_4/dropout_1/Mul_1Ğ
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall2model_4/sequential_4/dropout_4/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732A
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1Ħ
&model_4/sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_4/sequential_4/flatten_4/Const_1
(model_4/sequential_4/flatten_4/Reshape_1ReshapeHmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0/model_4/sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_4/sequential_4/flatten_4/Reshape_1ê
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpü
%model_4/sequential_4/dense_4/MatMul_1MatMul1model_4/sequential_4/flatten_4/Reshape_1:output:0<model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_4/sequential_4/dense_4/MatMul_1è
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpŝ
&model_4/sequential_4/dense_4/BiasAdd_1BiasAdd/model_4/sequential_4/dense_4/MatMul_1:product:0=model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_4/sequential_4/dense_4/BiasAdd_1Ħ
(model_4/distance_layer_4/PartitionedCallPartitionedCall-model_4/sequential_4/dense_4/BiasAdd:output:0/model_4/sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472*
(model_4/distance_layer_4/PartitionedCallÛ
IdentityIdentity1model_4/distance_layer_4/PartitionedCall:output:04^model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp6^model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp3^model_4/sequential_4/dense_4/MatMul/ReadVariableOp5^model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp2n
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp2h
2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2l
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2

f
G__inference_dropout_4_layer_call_and_return_conditional_losses_20979403

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ô@

E__inference_model_4_layer_call_and_return_conditional_losses_20978607

inputs
inputs_17
3sequential_4_dense_4_matmul_readvariableop_resource8
4sequential_4_dense_4_biasadd_readvariableop_resource
identity˘+sequential_4/dense_4/BiasAdd/ReadVariableOp˘-sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘*sequential_4/dense_4/MatMul/ReadVariableOp˘,sequential_4/dense_4/MatMul_1/ReadVariableOp
$sequential_4/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_4/dropout_4/dropout/Constı
"sequential_4/dropout_4/dropout/MulMulinputs-sequential_4/dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_4/dropout_4/dropout/Mul
$sequential_4/dropout_4/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2&
$sequential_4/dropout_4/dropout/Shapeú
;sequential_4/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_4/dropout_4/dropout/random_uniform/RandomUniform£
-sequential_4/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_4/dropout_4/dropout/GreaterEqual/y
+sequential_4/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_4/dropout_4/dropout/GreaterEqualĊ
#sequential_4/dropout_4/dropout/CastCast/sequential_4/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_4/dropout_4/dropout/Cast×
$sequential_4/dropout_4/dropout/Mul_1Mul&sequential_4/dropout_4/dropout/Mul:z:0'sequential_4/dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_4/dropout_4/dropout/Mul_1
5sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall(sequential_4/dropout_4/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817327
5sequential_4/one_hot_encoding_layer_4/PartitionedCall
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_4/flatten_4/Constċ
sequential_4/flatten_4/ReshapeReshape>sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_4/flatten_4/ReshapeÎ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOpÔ
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMulÌ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOpÖ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/BiasAdd
&sequential_4/dropout_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_4/dropout_4/dropout_1/ConstÁ
$sequential_4/dropout_4/dropout_1/MulMulinputs_1/sequential_4/dropout_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_4/dropout_4/dropout_1/Mul
&sequential_4/dropout_4/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_4/dropout_4/dropout_1/Shape
=sequential_4/dropout_4/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_4/dropout_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_4/dropout_4/dropout_1/random_uniform/RandomUniform§
/sequential_4/dropout_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_4/dropout_4/dropout_1/GreaterEqual/y£
-sequential_4/dropout_4/dropout_1/GreaterEqualGreaterEqualFsequential_4/dropout_4/dropout_1/random_uniform/RandomUniform:output:08sequential_4/dropout_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_4/dropout_4/dropout_1/GreaterEqualË
%sequential_4/dropout_4/dropout_1/CastCast1sequential_4/dropout_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_4/dropout_4/dropout_1/Castß
&sequential_4/dropout_4/dropout_1/Mul_1Mul(sequential_4/dropout_4/dropout_1/Mul:z:0)sequential_4/dropout_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_4/dropout_4/dropout_1/Mul_1
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall*sequential_4/dropout_4/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817329
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1
sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_4/flatten_4/Const_1í
 sequential_4/flatten_4/Reshape_1Reshape@sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0'sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_4/flatten_4/Reshape_1Ò
,sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_4/dense_4/MatMul_1/ReadVariableOpÜ
sequential_4/dense_4/MatMul_1MatMul)sequential_4/flatten_4/Reshape_1:output:04sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMul_1?
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpŜ
sequential_4/dense_4/BiasAdd_1BiasAdd'sequential_4/dense_4/MatMul_1:product:05sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_4/dense_4/BiasAdd_1
 distance_layer_4/PartitionedCallPartitionedCall%sequential_4/dense_4/BiasAdd:output:0'sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472"
 distance_layer_4/PartitionedCall³
IdentityIdentity)distance_layer_4/PartitionedCall:output:0,^sequential_4/dense_4/BiasAdd/ReadVariableOp.^sequential_4/dense_4/BiasAdd_1/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp-^sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2^
-sequential_4/dense_4/BiasAdd_1/ReadVariableOp-sequential_4/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2\
,sequential_4/dense_4/MatMul_1/ReadVariableOp,sequential_4/dense_4/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĝ
0
__inference_call_20978896
x
identityQ
CastCastx*

DstT0*

SrcT0* 
_output_shapes
:
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*$
_output_shapes
:2	
one_hota
IdentityIdentityone_hot:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
:
:C ?
 
_output_shapes
:


_user_specified_namex
Ĥ
?
cond_false_20978230
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
½
m
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_20978300
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ŝ@
Ħ
E__inference_model_4_layer_call_and_return_conditional_losses_20979035
inputs_0
inputs_17
3sequential_4_dense_4_matmul_readvariableop_resource8
4sequential_4_dense_4_biasadd_readvariableop_resource
identity˘+sequential_4/dense_4/BiasAdd/ReadVariableOp˘-sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘*sequential_4/dense_4/MatMul/ReadVariableOp˘,sequential_4/dense_4/MatMul_1/ReadVariableOp
$sequential_4/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_4/dropout_4/dropout/Constğ
"sequential_4/dropout_4/dropout/MulMulinputs_0-sequential_4/dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_4/dropout_4/dropout/Mul
$sequential_4/dropout_4/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_4/dropout_4/dropout/Shapeú
;sequential_4/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_4/dropout_4/dropout/random_uniform/RandomUniform£
-sequential_4/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_4/dropout_4/dropout/GreaterEqual/y
+sequential_4/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_4/dropout_4/dropout/GreaterEqualĊ
#sequential_4/dropout_4/dropout/CastCast/sequential_4/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_4/dropout_4/dropout/Cast×
$sequential_4/dropout_4/dropout/Mul_1Mul&sequential_4/dropout_4/dropout/Mul:z:0'sequential_4/dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_4/dropout_4/dropout/Mul_1
5sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall(sequential_4/dropout_4/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817327
5sequential_4/one_hot_encoding_layer_4/PartitionedCall
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_4/flatten_4/Constċ
sequential_4/flatten_4/ReshapeReshape>sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_4/flatten_4/ReshapeÎ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOpÔ
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMulÌ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOpÖ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/BiasAdd
&sequential_4/dropout_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_4/dropout_4/dropout_1/ConstÁ
$sequential_4/dropout_4/dropout_1/MulMulinputs_1/sequential_4/dropout_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_4/dropout_4/dropout_1/Mul
&sequential_4/dropout_4/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_4/dropout_4/dropout_1/Shape
=sequential_4/dropout_4/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_4/dropout_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_4/dropout_4/dropout_1/random_uniform/RandomUniform§
/sequential_4/dropout_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_4/dropout_4/dropout_1/GreaterEqual/y£
-sequential_4/dropout_4/dropout_1/GreaterEqualGreaterEqualFsequential_4/dropout_4/dropout_1/random_uniform/RandomUniform:output:08sequential_4/dropout_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_4/dropout_4/dropout_1/GreaterEqualË
%sequential_4/dropout_4/dropout_1/CastCast1sequential_4/dropout_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_4/dropout_4/dropout_1/Castß
&sequential_4/dropout_4/dropout_1/Mul_1Mul(sequential_4/dropout_4/dropout_1/Mul:z:0)sequential_4/dropout_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_4/dropout_4/dropout_1/Mul_1
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall*sequential_4/dropout_4/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_2097817329
7sequential_4/one_hot_encoding_layer_4/PartitionedCall_1
sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_4/flatten_4/Const_1í
 sequential_4/flatten_4/Reshape_1Reshape@sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0'sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_4/flatten_4/Reshape_1Ò
,sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_4/dense_4/MatMul_1/ReadVariableOpÜ
sequential_4/dense_4/MatMul_1MatMul)sequential_4/flatten_4/Reshape_1:output:04sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_4/dense_4/MatMul_1?
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_4/dense_4/BiasAdd_1/ReadVariableOpŜ
sequential_4/dense_4/BiasAdd_1BiasAdd'sequential_4/dense_4/MatMul_1:product:05sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_4/dense_4/BiasAdd_1
 distance_layer_4/PartitionedCallPartitionedCall%sequential_4/dense_4/BiasAdd:output:0'sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472"
 distance_layer_4/PartitionedCall³
IdentityIdentity)distance_layer_4/PartitionedCall:output:0,^sequential_4/dense_4/BiasAdd/ReadVariableOp.^sequential_4/dense_4/BiasAdd_1/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp-^sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2^
-sequential_4/dense_4/BiasAdd_1/ReadVariableOp-sequential_4/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2\
,sequential_4/dense_4/MatMul_1/ReadVariableOp,sequential_4/dense_4/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1

¸
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978361
input_5
dense_4_20978355
dense_4_20978357
identity˘dense_4/StatefulPartitionedCallŜ
dropout_4/PartitionedCallPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_209782782
dropout_4/PartitionedCallŞ
(one_hot_encoding_layer_4/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_209783002*
(one_hot_encoding_layer_4/PartitionedCall
flatten_4/PartitionedCallPartitionedCall1one_hot_encoding_layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_209783142
flatten_4/PartitionedCallµ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_20978355dense_4_20978357*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_209783322!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
?

*__inference_model_4_layer_call_fn_20979151
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_209785342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
î-
£
__inference_call_20978997
input_0
input_1?
;model_4_sequential_4_dense_4_matmul_readvariableop_resource@
<model_4_sequential_4_dense_4_biasadd_readvariableop_resource
identity˘3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp˘5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘2model_4/sequential_4/dense_4/MatMul/ReadVariableOp˘4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp
'model_4/sequential_4/dropout_4/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_4/sequential_4/dropout_4/Identity?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall0model_4/sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall
$model_4/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_4/sequential_4/flatten_4/Const
&model_4/sequential_4/flatten_4/ReshapeReshapeFmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0-model_4/sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_4/sequential_4/flatten_4/Reshapeĉ
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpô
#model_4/sequential_4/dense_4/MatMulMatMul/model_4/sequential_4/flatten_4/Reshape:output:0:model_4/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_4/sequential_4/dense_4/MatMulä
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpö
$model_4/sequential_4/dense_4/BiasAddBiasAdd-model_4/sequential_4/dense_4/MatMul:product:0;model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_4/sequential_4/dense_4/BiasAdd
)model_4/sequential_4/dropout_4/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_4/sequential_4/dropout_4/Identity_1Ğ
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall2model_4/sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209781732A
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1Ħ
&model_4/sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_4/sequential_4/flatten_4/Const_1
(model_4/sequential_4/flatten_4/Reshape_1ReshapeHmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0/model_4/sequential_4/flatten_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_4/sequential_4/flatten_4/Reshape_1ê
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpü
%model_4/sequential_4/dense_4/MatMul_1MatMul1model_4/sequential_4/flatten_4/Reshape_1:output:0<model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_4/sequential_4/dense_4/MatMul_1è
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpŝ
&model_4/sequential_4/dense_4/BiasAdd_1BiasAdd/model_4/sequential_4/dense_4/MatMul_1:product:0=model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_4/sequential_4/dense_4/BiasAdd_1Ħ
(model_4/distance_layer_4/PartitionedCallPartitionedCall-model_4/sequential_4/dense_4/BiasAdd:output:0/model_4/sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782472*
(model_4/distance_layer_4/PartitionedCallÛ
IdentityIdentity1model_4/distance_layer_4/PartitionedCall:output:04^model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp6^model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp3^model_4/sequential_4/dense_4/MatMul/ReadVariableOp5^model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp2n
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp2h
2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2l
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1

0
__inference_call_20979441
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
ß%
Ê
!__inference__traced_save_20979543
file_prefix$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardĤ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB-loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB-loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¤
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesï
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĦ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*^
_input_shapesM
K: : : : : : : : :
ô8::
ô8::
ô8:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
ô8:!	

_output_shapes	
::&
"
 
_output_shapes
:
ô8:!

_output_shapes	
::&"
 
_output_shapes
:
ô8:!

_output_shapes	
::

_output_shapes
: 
§
H
,__inference_flatten_4_layer_call_fn_20979461

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_209783142
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82

Identity"
identityIdentity:output:0*+
_input_shapes
:˙˙˙˙˙˙˙˙˙:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
"
9
__inference_call_20978247
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNotı
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *&
else_branchR
cond_false_20978230*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_209782292
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
Ċ
Ü
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978349
input_5
dense_4_20978343
dense_4_20978345
identity˘dense_4/StatefulPartitionedCall˘!dropout_4/StatefulPartitionedCallö
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_209782732#
!dropout_4/StatefulPartitionedCall²
(one_hot_encoding_layer_4/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_209783002*
(one_hot_encoding_layer_4/PartitionedCall
flatten_4/PartitionedCallPartitionedCall1one_hot_encoding_layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_209783142
flatten_4/PartitionedCallµ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_20978343dense_4_20978345*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_209783322!
dense_4/StatefulPartitionedCall?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
ó

/__inference_sequential_4_layer_call_fn_20978404
input_5
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_209783972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
î,
£
__inference_call_20978973
input_0
input_1?
;model_4_sequential_4_dense_4_matmul_readvariableop_resource@
<model_4_sequential_4_dense_4_biasadd_readvariableop_resource
identity˘3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp˘5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp˘2model_4/sequential_4/dense_4/MatMul/ReadVariableOp˘4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp
'model_4/sequential_4/dropout_4/IdentityIdentityinput_0*
T0* 
_output_shapes
:
2)
'model_4/sequential_4/dropout_4/Identity
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCallPartitionedCall0model_4/sequential_4/dropout_4/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209788962?
=model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall
$model_4/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_4/sequential_4/flatten_4/Constŭ
&model_4/sequential_4/flatten_4/ReshapeReshapeFmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall:output:0-model_4/sequential_4/flatten_4/Const:output:0*
T0* 
_output_shapes
:
ô82(
&model_4/sequential_4/flatten_4/Reshapeĉ
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_4/sequential_4/dense_4/MatMul/ReadVariableOpì
#model_4/sequential_4/dense_4/MatMulMatMul/model_4/sequential_4/flatten_4/Reshape:output:0:model_4/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2%
#model_4/sequential_4/dense_4/MatMulä
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOpî
$model_4/sequential_4/dense_4/BiasAddBiasAdd-model_4/sequential_4/dense_4/MatMul:product:0;model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$model_4/sequential_4/dense_4/BiasAdd
)model_4/sequential_4/dropout_4/Identity_1Identityinput_1*
T0* 
_output_shapes
:
2+
)model_4/sequential_4/dropout_4/Identity_1£
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1PartitionedCall2model_4/sequential_4/dropout_4/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209788962A
?model_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1Ħ
&model_4/sequential_4/flatten_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_4/sequential_4/flatten_4/Const_1
(model_4/sequential_4/flatten_4/Reshape_1ReshapeHmodel_4/sequential_4/one_hot_encoding_layer_4/PartitionedCall_1:output:0/model_4/sequential_4/flatten_4/Const_1:output:0*
T0* 
_output_shapes
:
ô82*
(model_4/sequential_4/flatten_4/Reshape_1ê
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_4_sequential_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOpô
%model_4/sequential_4/dense_4/MatMul_1MatMul1model_4/sequential_4/flatten_4/Reshape_1:output:0<model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2'
%model_4/sequential_4/dense_4/MatMul_1è
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_4_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOpö
&model_4/sequential_4/dense_4/BiasAdd_1BiasAdd/model_4/sequential_4/dense_4/MatMul_1:product:0=model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2(
&model_4/sequential_4/dense_4/BiasAdd_1
(model_4/distance_layer_4/PartitionedCallPartitionedCall-model_4/sequential_4/dense_4/BiasAdd:output:0/model_4/sequential_4/dense_4/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes	
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209789702*
(model_4/distance_layer_4/PartitionedCallÓ
IdentityIdentity1model_4/distance_layer_4/PartitionedCall:output:04^model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp6^model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp3^model_4/sequential_4/dense_4/MatMul/ReadVariableOp5^model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :
:
::2j
3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp3model_4/sequential_4/dense_4/BiasAdd/ReadVariableOp2n
5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp5model_4/sequential_4/dense_4/BiasAdd_1/ReadVariableOp2h
2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2model_4/sequential_4/dense_4/MatMul/ReadVariableOp2l
4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp4model_4/sequential_4/dense_4/MatMul_1/ReadVariableOp:I E
 
_output_shapes
:

!
_user_specified_name	input/0:IE
 
_output_shapes
:

!
_user_specified_name	input/1
â
9
cond_true_20979316
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yd
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*
_output_shapes	
:2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yd
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*
_output_shapes	
:2

cond/subR
	cond/SqrtSqrtcond/sub:z:0*
T0*
_output_shapes	
:2
	cond/Sqrt_
cond/IdentityIdentitycond/Sqrt:y:0*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:
Ú
ĥ
#__inference__wrapped_model_20978257
input_1
input_2
siamese_model_4_20978251
siamese_model_4_20978253
identity˘'siamese_model_4/StatefulPartitionedCall
'siamese_model_4/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2siamese_model_4_20978251siamese_model_4_20978253*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_call_209782502)
'siamese_model_4/StatefulPartitionedCallŞ
IdentityIdentity0siamese_model_4/StatefulPartitionedCall:output:0(^siamese_model_4/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2R
'siamese_model_4/StatefulPartitionedCall'siamese_model_4/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2

S
3__inference_distance_layer_4_layer_call_fn_20979277
s1
s2
identityÌ
PartitionedCallPartitionedCalls1s2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_209784922
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
ż 
9
__inference_call_20978970
s1
s2
identityD
subSubs1s2*
T0* 
_output_shapes
:
2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yU
powPowsub:z:0pow/y:output:0*
T0* 
_output_shapes
:
2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indices`
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yV
pow_1Pows2pow_1/y:output:0*
T0* 
_output_shapes
:
2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesh
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yV
pow_2Pows1pow_2/y:output:0*
T0* 
_output_shapes
:
2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesh
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/x]
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*
_output_shapes	
:2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Const^
MaximumMaximum	sub_1:z:0Const:output:0*
T0*
_output_shapes	
:2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/x]
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*
_output_shapes	
:2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1d
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*
_output_shapes	
:2
	Maximum_1S
mulMulMaximum:z:0Maximum_1:z:0*
T0*
_output_shapes	
:2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2b
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*
_output_shapes	
:2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x[
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*
_output_shapes	
:2
mul_1]
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*
_output_shapes	
:2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/xV
addAddV2add/x:output:0truediv:z:0*
T0*
_output_shapes	
:2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/yV
pow_3Powadd:z:0pow_3/y:output:0*
T0*
_output_shapes	
:2
pow_3H
IsInfIsInf	pow_3:z:0*
T0*
_output_shapes	
:2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNotİ
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes	
:* 
_read_only_resource_inputs
 *&
else_branchR
cond_false_20978953*
output_shapes	
:*%
then_branchR
cond_true_209789522
cond_
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes	
:2
cond/Identity^
add_1AddV2add:z:0cond/Identity:output:0*
T0*
_output_shapes	
:2
add_1B
LogLog	add_1:z:0*
T0*
_output_shapes	
:2
LogO
IdentityIdentityLog:y:0*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*+
_input_shapes
:
:
:D @
 
_output_shapes
:


_user_specified_names1:D@
 
_output_shapes
:


_user_specified_names2
¤9
Ò
$__inference__traced_restore_20979592
file_prefix
assignvariableop_total
assignvariableop_1_count 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate%
!assignvariableop_7_dense_4_kernel#
assignvariableop_8_dense_4_bias,
(assignvariableop_9_adam_dense_4_kernel_m+
'assignvariableop_10_adam_dense_4_bias_m-
)assignvariableop_11_adam_dense_4_kernel_v+
'assignvariableop_12_adam_dense_4_bias_v
identity_14˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB-loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB-loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesñ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_totalIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_countIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2Ħ
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5˘
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ş
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ĥ
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_4_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_4_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9­
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_dense_4_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ż
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_dense_4_bias_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ħ
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_4_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ż
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_dense_4_bias_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpü
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13ï
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

f
G__inference_dropout_4_layer_call_and_return_conditional_losses_20978273

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ô"
n
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_20979271
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNotı
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *&
else_branchR
cond_false_20979254*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_209792532
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
í

2__inference_siamese_model_4_layer_call_fn_20978801
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_209786822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2"ħL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ĉ
serving_defaultÒ
<
input_11
serving_default_input_1:0˙˙˙˙˙˙˙˙˙
<
input_21
serving_default_input_2:0˙˙˙˙˙˙˙˙˙8
output_1,
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:äë
Ì
siamese_network
loss_tracker
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
e__call__
*f&call_and_return_all_conditional_losses
g_default_save_signature
hcall"Ĉ
_tf_keras_modelĴ{"class_name": "SiameseModel", "name": "siamese_model_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SiameseModel"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 1, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ê

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_networkñ{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence1"}, "name": "sequence1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence2"}, "name": "sequence2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_4", "inbound_nodes": [[["sequence1", 0, 0, {}]], [["sequence2", 0, 0, {}]]]}, {"class_name": "DistanceLayer", "config": {"layer was saved without config": true}, "name": "distance_layer_4", "inbound_nodes": [[["sequential_4", 1, 0, {"s2": ["sequential_4", 2, 0]}]]]}], "input_layers": [["sequence1", 0, 0], ["sequence2", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["distance_layer_4", 0, 0]]}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1821]}, {"class_name": "TensorShape", "items": [null, 1821]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
ğ
	total
	count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
w
iter

beta_1

beta_2
	decay
learning_ratemambvcvd"
	optimizer
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ê
trainable_variables
metrics
layer_regularization_losses
regularization_losses

layers
	variables
 non_trainable_variables
!layer_metrics
e__call__
g_default_save_signature
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
,
kserving_default"
signature_map
ó"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "sequence1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence1"}}
ó"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "sequence2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence2"}}

"layer-0
#layer-1
$layer-2
%layer_with_weights-0
%layer-3
&trainable_variables
'regularization_losses
(	variables
)	keras_api
l__call__
*m&call_and_return_all_conditional_losses"¤
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1821]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
ı
*trainable_variables
+regularization_losses
,	variables
-	keras_api
n__call__
*o&call_and_return_all_conditional_losses
pcall" 
_tf_keras_layer{"class_name": "DistanceLayer", "name": "distance_layer_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
trainable_variables
.metrics
/layer_regularization_losses
regularization_losses

0layers
	variables
1non_trainable_variables
2layer_metrics
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
.
0
1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
": 
ô82dense_4/kernel
:2dense_4/bias
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
*
loss"
trackable_dict_wrapper
ċ
3trainable_variables
4regularization_losses
5	variables
6	keras_api
q__call__
*r&call_and_return_all_conditional_losses"Ö
_tf_keras_layerĵ{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
Ç
7trainable_variables
8regularization_losses
9	variables
:	keras_api
s__call__
*t&call_and_return_all_conditional_losses
ucall"?
_tf_keras_layer{"class_name": "OneHotEncodingLayer", "name": "one_hot_encoding_layer_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ĉ
;trainable_variables
<regularization_losses
=	variables
>	keras_api
v__call__
*w&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
÷

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
x__call__
*y&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7284}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7284]}}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
&trainable_variables
Cmetrics
Dlayer_regularization_losses
'regularization_losses

Elayers
(	variables
Fnon_trainable_variables
Glayer_metrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
*trainable_variables
Hmetrics
Ilayer_regularization_losses

Jlayers
+regularization_losses
,	variables
Knon_trainable_variables
Llayer_metrics
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
3trainable_variables
Mmetrics
Nlayer_regularization_losses

Olayers
4regularization_losses
5	variables
Pnon_trainable_variables
Qlayer_metrics
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
7trainable_variables
Rmetrics
Slayer_regularization_losses

Tlayers
8regularization_losses
9	variables
Unon_trainable_variables
Vlayer_metrics
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
;trainable_variables
Wmetrics
Xlayer_regularization_losses

Ylayers
<regularization_losses
=	variables
Znon_trainable_variables
[layer_metrics
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
?trainable_variables
\metrics
]layer_regularization_losses

^layers
@regularization_losses
A	variables
_non_trainable_variables
`layer_metrics
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
':%
ô82Adam/dense_4/kernel/m
 :2Adam/dense_4/bias/m
':%
ô82Adam/dense_4/kernel/v
 :2Adam/dense_4/bias/v
2
2__inference_siamese_model_4_layer_call_fn_20978883
2__inference_siamese_model_4_layer_call_fn_20978791
2__inference_siamese_model_4_layer_call_fn_20978873
2__inference_siamese_model_4_layer_call_fn_20978801²
İ²?
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ñ
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978757
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978781
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978839
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978863²
İ²?
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
#__inference__wrapped_model_20978257à
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *P˘M
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
Ŭ2Ú
__inference_call_20978973
__inference_call_20978997Ħ
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Î2Ë
*__inference_model_4_layer_call_fn_20979069
*__inference_model_4_layer_call_fn_20979161
*__inference_model_4_layer_call_fn_20979151
*__inference_model_4_layer_call_fn_20979079
*__inference_model_4_layer_call_fn_20978565
*__inference_model_4_layer_call_fn_20978541À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
?2í
E__inference_model_4_layer_call_and_return_conditional_losses_20979059
E__inference_model_4_layer_call_and_return_conditional_losses_20978502
E__inference_model_4_layer_call_and_return_conditional_losses_20979141
E__inference_model_4_layer_call_and_return_conditional_losses_20979035
E__inference_model_4_layer_call_and_return_conditional_losses_20978516
E__inference_model_4_layer_call_and_return_conditional_losses_20979117À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ÔBÑ
&__inference_signature_wrapper_20978719input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
/__inference_sequential_4_layer_call_fn_20979205
/__inference_sequential_4_layer_call_fn_20978404
/__inference_sequential_4_layer_call_fn_20979214
/__inference_sequential_4_layer_call_fn_20978383À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ö2ó
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978361
J__inference_sequential_4_layer_call_and_return_conditional_losses_20979182
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978349
J__inference_sequential_4_layer_call_and_return_conditional_losses_20979196À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ß2Ü
3__inference_distance_layer_4_layer_call_fn_20979277¤
²
FullArgSpec
args
jself
js1
js2
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ú2÷
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_20979271¤
²
FullArgSpec
args
jself
js1
js2
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
à2Ŭ
__inference_call_20979391
__inference_call_20979334¤
²
FullArgSpec
args
jself
js1
js2
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
,__inference_dropout_4_layer_call_fn_20979418
,__inference_dropout_4_layer_call_fn_20979413´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ì2É
G__inference_dropout_4_layer_call_and_return_conditional_losses_20979408
G__inference_dropout_4_layer_call_and_return_conditional_losses_20979403´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
à2Ŭ
;__inference_one_hot_encoding_layer_4_layer_call_fn_20979432
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
û2ĝ
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_20979427
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ù2Ö
__inference_call_20979441
__inference_call_20979450
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
,__inference_flatten_4_layer_call_fn_20979461˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ñ2î
G__inference_flatten_4_layer_call_and_return_conditional_losses_20979456˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ñ
*__inference_dense_4_layer_call_fn_20979480˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ï2ì
E__inference_dense_4_layer_call_and_return_conditional_losses_20979471˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ı
#__inference__wrapped_model_20978257Z˘W
P˘M
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
Ş "/Ş,
*
output_1
output_1˙˙˙˙˙˙˙˙˙
__inference_call_20978973fJ˘G
@˘=
;˘8

input/0


input/1

Ş "˘

0
__inference_call_20978997~Z˘W
P˘M
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
Ş "˘

0˙˙˙˙˙˙˙˙˙h
__inference_call_20979334K;˘8
1˘.

s1


s2

Ş "	
__inference_call_20979391cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙i
__inference_call_20979441L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Y
__inference_call_20979450<#˘ 
˘

x

Ş "§
E__inference_dense_4_layer_call_and_return_conditional_losses_20979471^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
*__inference_dense_4_layer_call_fn_20979480Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "˙˙˙˙˙˙˙˙˙Â
N__inference_distance_layer_4_layer_call_and_return_conditional_losses_20979271pK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 
3__inference_distance_layer_4_layer_call_fn_20979277cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙İ
G__inference_dropout_4_layer_call_and_return_conditional_losses_20979403^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 İ
G__inference_dropout_4_layer_call_and_return_conditional_losses_20979408^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
,__inference_dropout_4_layer_call_fn_20979413Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙
,__inference_dropout_4_layer_call_fn_20979418Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙İ
G__inference_flatten_4_layer_call_and_return_conditional_losses_20979456^4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ô8
 
,__inference_flatten_4_layer_call_fn_20979461Q4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ô8à
E__inference_model_4_layer_call_and_return_conditional_losses_20978502f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 à
E__inference_model_4_layer_call_and_return_conditional_losses_20978516f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p 

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŝ
E__inference_model_4_layer_call_and_return_conditional_losses_20979035d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŝ
E__inference_model_4_layer_call_and_return_conditional_losses_20979059d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŝ
E__inference_model_4_layer_call_and_return_conditional_losses_20979117d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŝ
E__inference_model_4_layer_call_and_return_conditional_losses_20979141d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ı
*__inference_model_4_layer_call_fn_20978541f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0˙˙˙˙˙˙˙˙˙ı
*__inference_model_4_layer_call_fn_20978565f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0˙˙˙˙˙˙˙˙˙·
*__inference_model_4_layer_call_fn_20979069d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0˙˙˙˙˙˙˙˙˙·
*__inference_model_4_layer_call_fn_20979079d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0˙˙˙˙˙˙˙˙˙·
*__inference_model_4_layer_call_fn_20979151d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0˙˙˙˙˙˙˙˙˙·
*__inference_model_4_layer_call_fn_20979161d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0˙˙˙˙˙˙˙˙˙³
V__inference_one_hot_encoding_layer_4_layer_call_and_return_conditional_losses_20979427Y+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 
;__inference_one_hot_encoding_layer_4_layer_call_fn_20979432L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙µ
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978349g9˘6
/˘,
"
input_5˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 µ
J__inference_sequential_4_layer_call_and_return_conditional_losses_20978361g9˘6
/˘,
"
input_5˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ´
J__inference_sequential_4_layer_call_and_return_conditional_losses_20979182f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ´
J__inference_sequential_4_layer_call_and_return_conditional_losses_20979196f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
/__inference_sequential_4_layer_call_fn_20978383Z9˘6
/˘,
"
input_5˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_4_layer_call_fn_20978404Z9˘6
/˘,
"
input_5˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_4_layer_call_fn_20979205Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_4_layer_call_fn_20979214Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙à
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978757^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 à
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978781^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 à
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978839^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 à
M__inference_siamese_model_4_layer_call_and_return_conditional_losses_20978863^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ı
2__inference_siamese_model_4_layer_call_fn_20978791^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p
Ş "˘

0˙˙˙˙˙˙˙˙˙ı
2__inference_siamese_model_4_layer_call_fn_20978801^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0˙˙˙˙˙˙˙˙˙ı
2__inference_siamese_model_4_layer_call_fn_20978873^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p
Ş "˘

0˙˙˙˙˙˙˙˙˙ı
2__inference_siamese_model_4_layer_call_fn_20978883^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0˙˙˙˙˙˙˙˙˙Í
&__inference_signature_wrapper_20978719˘k˘h
˘ 
aŞ^
-
input_1"
input_1˙˙˙˙˙˙˙˙˙
-
input_2"
input_2˙˙˙˙˙˙˙˙˙"/Ş,
*
output_1
output_1˙˙˙˙˙˙˙˙˙