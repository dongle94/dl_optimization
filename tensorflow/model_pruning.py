import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model_path = '/data/dl_optimization/tensorflow/model/falldown-cls-2.h5'
prune_model_path = '/data/dl_optimization/tensorflow/model/test_effidet_do/pruning_model.h5'

model = keras.models.load_model(model_path)
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
# load dataset
batch_size = 32
train_datasets = keras.utils.image_dataset_from_directory(
    '/data/dl_optimization/tensorflow/data/falldown/train',
    labels='inferred',
    label_mode='categorical',
    class_names=['falldown', 'no_falldown'],
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True,
    seed=35,
    validation_split=0.1,
    subset='training',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

test_datasets = keras.utils.image_dataset_from_directory(
    '/data/dl_optimization/tensorflow/data/falldown/test',
    labels='inferred',
    label_mode='categorical',
    class_names=['falldown', 'no_falldown'],
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# write base model evaluation
_, baseline_model_accuracy = model.evaluate(
    x=test_datasets,
    batch_size=32,
    verbose=1)
print('Baseline test accuracy:', baseline_model_accuracy)
exit()
# pruning task
epochs = 1
end_stop = len(train_datasets) * epochs
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_stop)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
print(model_for_pruning.summary())


# pruning callback
logdir = './logs'
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
model_for_pruning.fit(
    x=train_datasets,
    batch_size=batch_size,
    epochs=epochs,
    #validation_split=validation_split,
    callbacks=callbacks,
    verbose=1)

# test pruning model
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
    x=test_datasets,
    batch_size=32,
    verbose=1)
print('Baseline test accuracy:', baseline_model_accuracy)
print('Pruned test accuracy:', model_for_pruning_accuracy)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
tf.keras.models.save_model(model_for_export, './model/test_effidet_do/pruning_save_model.h5', include_optimizer=True)
model_for_export.save('./model/test_effidet_do/pruning_model.h5')
print(type(model))
print(type(model_for_pruning))
print(type(model_for_export))