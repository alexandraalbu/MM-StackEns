from tensorflow.keras.layers import Layer, Dense


class IndependentGaussian(Layer):

    def __init__(self, event_shape, std_activation, **kwargs):
        super().__init__(**kwargs)

        self._mean = Dense(units=event_shape)
        # since we use a diagonal covariance matrix, we can output directly the standard deviation vector
        # (this simplifies our future computations)
        self._std = Dense(units=event_shape, activation=std_activation)

    def call(self, inputs, **kwargs):
        return [self._mean(inputs), self._std(inputs)]

