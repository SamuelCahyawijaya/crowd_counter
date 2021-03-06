def create_left_branch(input_layer, kernel_size=(3, 3)):
  x = Conv2D(64, kernel_size, padding='same')(input_layer)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=1)(x)
  x = Dropout(0.25)(x)

  x = Conv2D(128, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=1)(x)
  x = Dropout(0.25)(x)

  x = Conv2D(256, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=1)(x)
  x = Dropout(0.5)(x)

  x = Conv2D(512, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=1)(x)
  x = Dropout(0.5)(x)

  return x

def create_right_branch(input_layer, kernel_size=(5, 5)):
  x = Conv2D(24, kernel_size, padding='same')(input_layer)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = AveragePooling2D(pool_size=(5, 5), padding='same', strides=1)(x)

  x = Conv2D(24, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = AveragePooling2D(pool_size=(5, 5), padding='same', strides=1)(x)

  x = Conv2D(24, kernel_size, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = AveragePooling2D(pool_size=(5, 5), padding='same', strides=1)(x)

  return x

def create_model(batch_size=64, input_shape=(225, 225, 3)):
  input_layer = Input(shape=input_shape)
  left_branch = create_left_branch(input_layer)
  right_branch = create_right_branch(input_layer)
    
  x = Concatenate()([left_branch, right_branch])
  x = Conv2D(1, (1, 1), padding='same')(x)
  model = Model(inputs=input_layer, outputs=x)
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model


class CheckPoints(Callback):
  def on_train_begin(self, logs={}):
    self.epoch_nmb = 0
    return
 
  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    path = "weights"
    if not os.path.exists(path):
        os.makedirs(path)
    if self.epoch_nmb % 100 == 0:
      self.model.save("{}/step_{}.h5".format(path, self.epoch_nmb))
    self.epoch_nmb += 1
    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return