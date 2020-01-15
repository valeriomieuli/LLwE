'''import tensorflow.keras as keras
import tensorflow.keras.applications as applications


def build_features_extractor(model_name, input_shape):
    if model_name == 'VGG16':
        model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'VGG19':
        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        assert (False), "Specified base model is not available !"

    x = model.output
    x = keras.layers.Flatten()(x)  # keras.layers.GlobalMaxPooling2D()(x)

    model = keras.Model(inputs=model.input, outputs=x)
    return model


def build_autoencoder(n_input_features, hidden_layer_size, weight_decay):
    input = keras.layers.Input(shape=(n_input_features,))
    x = keras.layers.Dense(units=hidden_layer_size, activation=None,
                           kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Dense(units=n_input_features, activation=None,
                           kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Activation(activation='sigmoid')(x)

    model = keras.Model(inputs=input, outputs=x)
    return model


def build_expert(model_name, input_shape, n_classes, weight_decay):
    if model_name == 'ResNet152V2':
        model = applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape)
    elif model_name == 'InceptionResNetV2':
        model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                   input_shape=input_shape)
    elif model_name == 'ResNet50':
        model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError("Specified base model is not available !")

    x = model.output
    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(units=n_classes, activation='softmax')(x)
    model = keras.Model(inputs=model.input, outputs=x)

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = keras.regularizers.l2(weight_decay)

    return model'''
import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras import layers


def build_features_extractor(model_name, input_shape):
    if model_name == 'VGG16':
        model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'VGG19':
        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        assert (False), "Specified base model is not available !"

    x = model.output
    x = layers.Flatten()(x)
    return keras.Model(inputs=model.input, outputs=x)


def build_autoencoder(n_input_features, hidden_layer_size, weight_decay):
    input = layers.Input(shape=(n_input_features,))
    x = layers.Dense(units=hidden_layer_size, activation=None, )(input)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dense(units=n_input_features, activation=None)(x)
    x = layers.Activation(activation='sigmoid')(x)

    model = keras.Model(inputs=input, outputs=x)
    if weight_decay != -1:
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = keras.regularizers.l2(weight_decay)

    return model


def get_preprocessing_function(model_name):
    if model_name == 'InceptionResNetV2':
        return applications.inception_resnet_v2.preprocess_input
    elif model_name == 'VGG16':
        return applications.vgg16.preprocess_input
    elif model_name == 'VGG19':
        return applications.vgg19.preprocess_input
    elif model_name in ['ResNet50', 'ResNet18']:
        return applications.resnet50.preprocess_input
    else:
        raise ValueError("Preprocessing function for the specified base model is not available !")


def build_expert(model_name, input_shape, n_classes, weight_decay):
    if model_name == 'InceptionResNetV2':
        base_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                        input_shape=input_shape)
        head_model = layers.GlobalAveragePooling2D()(base_model.output)
        head_model = layers.Dense(units=n_classes, activation="softmax")(head_model)

    elif model_name == 'VGG16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        head_model = layers.Flatten()(base_model.output)
        head_model = layers.Dense(units=1024, activation='relu')(head_model)
        head_model = layers.Dense(units=1024, activation='relu')(head_model)
        head_model = layers.Dense(units=n_classes, activation='softmax')(head_model)

    elif model_name == 'VGG19':
        base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        head_model = layers.Flatten()(base_model.output)
        head_model = layers.Dense(units=1024)(head_model)
        head_model = layers.Dense(units=1024)(head_model)
        head_model = layers.Dense(units=n_classes, activation='softmax')(head_model)
    else:
        raise ValueError("Specified base model is not available !")

    model = keras.Model(inputs=base_model.input, outputs=head_model)
    if weight_decay != -1:
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = keras.regularizers.l2(weight_decay)

    return model
