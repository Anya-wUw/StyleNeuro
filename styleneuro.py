from xml.etree.ElementInclude import include
from grpc import Channel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from io import BytesIO
from PIL import Image 

img = Image.open('original.jpg')
img_style = Image.open('style.jpg')

#выводим загруженные фото
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_style)
plt.show()

#подготавливаем фото к обучению на vgg19
x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

# content_path = 'original.jpg'
# style_path = 'style.jpg'

#from BGR to RGB
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        #убираем добавленную 0ю ось
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    #обратно возвращаем значения RGB
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    #возвращаем изображение
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#вспомогательные переменные
#слой vgg19 для фильтрации по ориг. фото
content_layers = ['block5_conv2'] 

#слои vgg19 для фильтрации по стил. фото
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
#считаем кол-во получившихся слоев
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

#подгружаем обученную модель(по имеджнету) vgg19 без полносвязной сети
vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
#нельзя менять веса(не переобучать их)
vgg.trainable = False

#проходим по спискам, заполняя выходы
content_outputs = [vgg.get_layer(name).output for name in content_layers]
style_outputs = [vgg.get_layer(name).output for name in style_layers]
model_outputs = style_outputs + content_outputs

#выводим входы и выходы
print(vgg.input)
for m in model_outputs:
    print(m)

#выводим вывод структуры нейронки в консоль
model = keras.models.Model(vgg.input, model_outputs)
print(model.summary())

#потери по контенту Jc
#(оригинал-получивш.изображение)
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

#вычисление матрицы Грама
#input_tensor = nH*nW*nC(число каналов(карт признаков))
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1]) #nC
    a = tf.reshape(input_tensor, [-1, channels]) #преобразование в 2мерный тензор (nH*nW=G и nC)
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True) #G траспон.* G
    return gram / tf.cast(n, tf.float32) # 1/n * Gt * G


#потери по стилю (не по всем слоям, а для определенного = Js(l), l - слой(l=1,2..5))
#(фото(матрица) стиля-матрица Грама)
# Js(l)=1/(nC^2*(nH*nW)^2)* red.mean(Gp-Gs)^2
def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)

#карта признаков для стилей и контента
def get_feature_representations(model):
    # batch compute content and style features
    style_outputs = model(x_style)
    content_outputs = model(x_img)

    # Get the style and content feature representations from our model  
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

#вычисление всех потерь
#a-style_weight, b-content_weight; J = aJc + bJs
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and 
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    #карты признаков
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        #Js = sum(l=1..5)(g(l)*Js(l))
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    #J = aJc + bJs
    style_score *= style_weight
    content_score *= content_weight

    #находим общие потери
    loss = style_score + content_score 
    #возвращаем общие потери, потери по стилю, потери по контенту
    return loss, style_score, content_score

#количество итераций
num_iterations=100
#насколько важен контент (a) 1e3
content_weight=1e-1
#насколько важен стиль (b) 1e-2
style_weight=1e3

#заранее посчитаем карты признаков и матрицу Грама
style_features, content_features = get_feature_representations(model)
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

#начальное формируемое изображение = ориг.фото
init_image = np.copy(x_img)
init_image = tf.Variable(init_image, dtype=tf.float32)

#опимизатор для градиентного спуска
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
iter_count = 1
best_loss, best_img = float('inf'), None
loss_weights = (style_weight, content_weight)

cfg = {
    'model': model,
    'loss_weights': loss_weights,
    'init_image': init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}

#из BGR в RGB
norm_means = np.array([103.99, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means
imgs = []

#запуск град.спуска
for i in range(num_iterations):
    #вычисляем градиент
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)

    total_loss = all_loss[0]
    #градиент относительно пикселей изображения
    grads = tape.gradient(total_loss, init_image)

    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    
    #находим изображение с наим. потерями
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    #сохраняем лучшее изображение
    plot_img = init_image.numpy()
    plot_img = deprocess_img(plot_img)
    imgs.append(plot_img)
    print('Iteration: {}'.format(i))

plt.imshow(best_img)
print(best_loss)
plt.show()

image = Image.fromarray(best_img.astype('uint8'), 'RGB')
image.save("resalt.jpg")
