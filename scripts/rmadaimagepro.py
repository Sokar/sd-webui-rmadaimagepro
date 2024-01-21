from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from modules import scripts, script_callbacks
import pywt
import random
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
from translate import Translator
from langdetect import detect
import re
import numpy as np
import gradio as gr
import torch
from transformers import MarianMTModel, MarianTokenizer


model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


CONFIG_PATH = Path(__file__).parent.resolve() / '../config.yaml'



print(
    f"[-] rMada ProImage initialized."
)


class RmadaUPS(scripts.Script):
    def __init__(self):
        super().__init__()
        try:
            self.config: DictConfig = OmegaConf.load(CONFIG_PATH)
        except Exception:
            self.config = DictConfig({})
        self.disable = False
        self.step_limit = 0
        self.infotext_fields = []
        last_text = ""
        nuevo_texto = ""
        texto_traducido = []

    def title(self):
        return "rMada ProImage"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='rMada ProImage', open=False):
            with gr.Row():
                RMADA_enable = gr.Checkbox(label='Enable extension', value=self.config.get('RMADA_enable', False))
            with gr.Tab("Sharpen"):
                with gr.Row():
                    RMADA_sharpenweight = gr.Slider(minimum=0, maximum=10, step=0.1, label="Sharpen", value=self.config.get('RMADA_sharpenweight', 0))
                    RMADA_edge_detection_sharpening = gr.Slider(minimum=0, maximum=10, step=0.1, label="Edge Detection Sharpening", value=self.config.get('RMADA_edge_detection_sharpening', 0))
                with gr.Row():
                    RMADA_wavelet_sharpening = gr.Slider(minimum=0, maximum=5, step=0.1, label="Wavelet Sharpening", value=self.config.get('RMADA_wavelet_sharpening', 0))
                    RMADA_adaptive_sharpened =  gr.Slider(minimum=0, maximum=10, step=1, label="Adaptive Sharpen", value=self.config.get('RMADA_adaptive_sharpened', 0))
            with gr.Tab("Enhance"):
                with gr.Row():
                    RMADA_contrast = gr.Slider(minimum=-1, maximum=1, step=0.05, label="Contrast", value=self.config.get('RMADA_contrast', 0))
                    RMADA_brightness = gr.Slider(minimum=-1, maximum=1, step=0.05, label="Brightness", value=self.config.get('RMADA_brightness', 0))
                with gr.Row():
                    RMADA_saturation = gr.Slider(minimum=-1, maximum=1, step=0.05, label="Saturation", value=self.config.get('RMADA_saturation', 0))
            with gr.Tab("Filters"):
                with gr.Row():
                    RMADA_noise = gr.Slider(minimum=0, maximum=0.1, step=0.001, label="Noise", value=self.config.get('RMADA_noise', 0))
                with gr.Row():
                    RMADA_vignette = gr.Slider(minimum=0, maximum=1, step=0.01, label="Vignette", value=self.config.get('RMADA_vignette', 0))
                    RMADA_vignette_overlay = gr.Slider(minimum=0, maximum=1, step=0.01, label="Vignette Overlay", value=self.config.get('RMADA_vignette_overlay', 0))
            with gr.Tab("Utils"):
                with gr.Row():
                    RMADA_translate = gr.Checkbox(label='Auto Translate', value=self.config.get('RMADA_translate', False))
                    RMADA_translate_lang = gr.Dropdown(['es'], label='Lang From', 
                        value=self.config.get('RMADA_translate_lang', 'es'))
                with gr.Row():
                    RMADA_removeloras = gr.Checkbox(label='Remove all Loras from prompt', value=self.config.get('RMADA_removeloras', False))

        ui = [RMADA_enable, RMADA_sharpenweight, RMADA_edge_detection_sharpening, RMADA_wavelet_sharpening, RMADA_adaptive_sharpened, RMADA_contrast, RMADA_brightness, RMADA_saturation, RMADA_noise, RMADA_vignette, RMADA_vignette_overlay, RMADA_translate, RMADA_translate_lang, RMADA_removeloras]
        for elem in ui:
            setattr(elem, "do_not_save_to_config", True)

        parameters = {
            'RMADA_enable': RMADA_enable,
            'RMADA_sharpenweight': RMADA_sharpenweight,
            'RMADA_edge_detection_sharpening': RMADA_edge_detection_sharpening,
            'RMADA_wavelet_sharpening': RMADA_wavelet_sharpening,
            'RMADA_adaptive_sharpened': RMADA_adaptive_sharpened,
            'RMADA_contrast': RMADA_contrast,
            'RMADA_brightness': RMADA_brightness,
            'RMADA_saturation': RMADA_saturation,
            'RMADA_noise': RMADA_noise,
            'RMADA_vignette': RMADA_vignette,
            'RMADA_vignette_overlay': RMADA_vignette_overlay,
            'RMADA_translate': RMADA_translate,
            'RMADA_translate_lang': RMADA_translate_lang,
            'RMADA_removeloras': RMADA_removeloras
        }
        for k, element in parameters.items():
           self.infotext_fields.append((element, k))

        return ui

    def detectar_idioma(self, texto):
        return detect(texto)

    def traducir_segmentos(self, texto, seeds, lang):
        
        #traductor = Translator()
        # traductor = Translator(to_lang="en")
        translate_final = False
        last_text = ''
        nuevo_texto = texto[0]
        res = []
        for i, text in enumerate(texto):
            if last_text != text: 
                gen = random.Random()
                gen.seed(seeds[i])

                segmentos = re.split(r'([,.():])', text)  # Dividir el texto en segmentos
                texto_traducido = []

                for segmento in segmentos:
                    if segmento not in [',', '.', '(', ')', ':'] and not re.search(r'__.*?__', segmento) and re.search(r'[a-zA-Z]{2,}', segmento):  # Ignorar los separadores
                        deteccion = self.detectar_idioma(segmento)
                        if deteccion != 'en':  # Traducir solo si es español
                            traductor = Translator(from_lang=lang, to_lang='en')
                            old_segmento = segmento
                            segmentoParser = traductor.translate(segmento)
                            #if "MYMEMORY WARNING" not in segmentoParser:
                            #    segmento = segmentoParser
                            #    print("RMADA Translate: from ", old_segmento, " to ", segmento)
                            #else:
                            
                            # Texto de origen y destino
                            src_texts = segmento
                            tgt_texts = None  # No es necesario para la traducción

                            # Preparar los datos para el modelo
                            model_inputs = tokenizer(src_texts, return_tensors="pt", truncation=True, padding=True)
                            labels = tokenizer(text_target=tgt_texts, return_tensors="pt", truncation=True, padding=True) if tgt_texts else None
                            if labels:
                                model_inputs["labels"] = labels["input_ids"]

                            # Traducir
                            translated = model.generate(**model_inputs, max_length=512)
                            segmentoParser = tokenizer.decode(translated[0], skip_special_tokens=True)

                            segmento = segmentoParser
                            print("RMADA Translate: from ", old_segmento, " to ", segmento)
                                
                            translate_final = True
                    #texto_traducido.append(segmento)
                    # Añadir un espacio antes del '(' si no es el primer elemento
                    if segmento == '(' and texto_traducido:
                        texto_traducido.append(' ' + segmento)
                    elif segmento == ')' and texto_traducido:
                        texto_traducido.append(segmento + ' ')
                    else:
                        texto_traducido.append(segmento)

                res.append(''.join(texto_traducido))
                nuevo_texto = ''.join(texto_traducido)
                last_text = text
            else: 
                gen = random.Random()
                gen.seed(seeds[i])
                #print(nuevo_texto)
                res.append(''.join(nuevo_texto))

        if translate_final == True:
            return res
        
        return texto

        
    def removeLoras(self, texto):
        res = []
        for text in texto:
            # Eliminar la etiqueta <lora:XXXX:X>
            text = re.sub(r'<lora:[^>]*>', '', text)

            # Limpiar espacios y comas extra
            text = re.sub(r',\s*,', ',', text)  # Elimina comas duplicadas
            text = text.strip(', ')  # Elimina comas y espacios al inicio y al final

            res.append(text)
        return res
    
    def process(self, p, RMADA_enable, RMADA_sharpenweight, RMADA_edge_detection_sharpening, RMADA_wavelet_sharpening, RMADA_adaptive_sharpened, RMADA_contrast, RMADA_brightness, RMADA_saturation, RMADA_noise, RMADA_vignette, RMADA_vignette_overlay, RMADA_translate, RMADA_translate_lang, RMADA_removeloras):
        self.config = DictConfig({name: var for name, var in locals().items() if name not in ['self', 'p']})
        self.step_limit = 0

        if not RMADA_enable or self.disable:
            script_callbacks.remove_current_script_callbacks()
            return
        
        model = p.sd_model.model.diffusion_model
        
        if RMADA_removeloras:
            p.all_prompts = self.removeLoras(p.all_prompts)
        
        if RMADA_translate:
            original_prompt = p.all_prompts[0]
            # p.all_prompts[0] = self.traducir_segmentos(original_prompt)
            translate_prompt = self.traducir_segmentos(p.all_prompts, p.all_seeds, RMADA_translate_lang)
            if p.all_prompts != translate_prompt: # ya no es necesario, porque se encarga de verificar si es el mismo prompt la función de traducir
                p.all_prompts = translate_prompt
            if getattr(p, 'all_hr_prompts', None) is not None:
                p.all_hr_prompts = self.traducir_segmentos(p.all_hr_prompts, p.all_seeds, RMADA_translate_lang)
            #if original_prompt != p.all_prompts[0]:
            #    p.extra_generation_params["RMADA_Translate_Original"] = original_prompt

        parameters = {
            'RMADA_enable': RMADA_enable,
            'RMADA_sharpenweight': RMADA_sharpenweight,
            'RMADA_edge_detection_sharpening': RMADA_edge_detection_sharpening,
            'RMADA_wavelet_sharpening': RMADA_wavelet_sharpening,
            'RMADA_adaptive_sharpened': RMADA_adaptive_sharpened,
            'RMADA_contrast': RMADA_contrast,
            'RMADA_brightness': RMADA_brightness,
            'RMADA_saturation': RMADA_saturation,
            'RMADA_noise': RMADA_noise,
            'RMADA_vignette': RMADA_vignette,
            'RMADA_vignette_overlay': RMADA_vignette_overlay,
            'RMADA_translate': RMADA_translate,
            'RMADA_translate_lang': RMADA_translate_lang
        }
        for k, v in parameters.items():
            if v != 0 and 'RMADA_enable' not in k and 'RMADA_translate_lang' not in k and 'RMADA_translate' not in k and 'RMADA_removeloras' not in k:
                p.extra_generation_params[k] = v

    
    def postprocess(self, p, processed, *args):
        
            
        def apply_custom_sharpen(image, intensity=1.0):
            sharpen_matrix = [
                -1, -1, -1,
                -1, 9 + (10 - intensity), -1,
                -1, -1, -1
            ]
            sharpen_filter = ImageFilter.Kernel((3, 3), sharpen_matrix)
            return image.filter(sharpen_filter)
        
        def apply_unsharp_mask(image, radius=2, percent=150, threshold=3):
            return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

        def apply_high_pass_sharpen(image, blur_radius=2, scale=1.0):
            """
            Aplica un filtro de paso alto para el enfoque.

            :param image: Imagen PIL a procesar.
            :param blur_radius: Radio de desenfoque para la versión suavizada de la imagen.
            :param scale: Escala de enfoque aplicada a la imagen de paso alto.
            :return: Imagen PIL enfocada.
            """
            # Asegurarse de procesar la imagen en color
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Crear una versión suavizada de la imagen
            blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))

            # Crear la imagen de paso alto restando la versión suavizada de la imagen original
            high_pass = ImageChops.subtract(image, blurred_image)

            # Escalar la imagen de paso alto (opcional)
            high_pass = ImageChops.multiply(high_pass, scale)

            # Sumar la imagen de paso alto a la imagen original para enfocar
            sharpened_image = ImageChops.add(image, high_pass)

            return sharpened_image

        def apply_edge_detection_sharpen(image, intensity=1.0):
            """
            Aplica un enfoque de detección de bordes a la imagen.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad del enfoque.
            :return: Imagen PIL enfocada.
            """
            
            edge_threshold = 100
            
            # Asegurarse de procesar la imagen en color
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convertir a escala de grises para la detección de bordes
            gray_image = image.convert('L')

            # Aplicar el operador Sobel para la detección de bordes
            edges = gray_image.filter(ImageFilter.FIND_EDGES)

            # Convertir los bordes a una imagen en blanco y negro
            edges_bw = edges.point(lambda x: 0 if x < edge_threshold else 255, '1')

            # Convertir la imagen de bordes a un array y escalarla
            edge_data = np.array(edges_bw) / 255 * intensity

            # Ampliar edge_data para que tenga 3 canales
            edge_data = np.stack((edge_data, edge_data, edge_data), axis=-1)

            # Aplicar el enfoque a la imagen original basado en los bordes detectados
            original_data = np.array(image)
            
            # sharpened_data = np.clip(original_data + edge_data, 0, 255).astype(np.uint8)
            sharpened_data = np.clip(original_data + (edge_data * 255 * intensity), 0, 255).astype(np.uint8)

            # Convertir de nuevo a imagen PIL
            sharpened_image = Image.fromarray(sharpened_data)

            return sharpened_image
                
      
        def apply_wavelet_sharpen(image, intensity=0.5):
            """
            Aplica wavelet sharpening a una imagen y la fusiona con la original,
            manteniendo el espacio de color correcto.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad del enfoque.
            :return: Imagen PIL enfocada.
            """
            # Asegurarse de procesar la imagen en color
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convertir la imagen a un array de NumPy
            img_array = np.array(image)

            # Aplicar transformada wavelet por separado en cada canal de color
            final_sharpened_arrays = []
            for channel in range(3):  # Procesar R, G y B independientemente
                # Descomponer la imagen utilizando transformada wavelet
                coeffs = pywt.wavedec2(img_array[:, :, channel], 'haar', level=1)
                cA, (cH, cV, cD) = coeffs

                # Aplicar enfoque a los coeficientes de alta frecuencia
                cH += cH * intensity
                cV += cV * intensity
                cD += cD * intensity

                # Reconstruir la imagen a partir de los coeficientes modificados
                coeffs = cA, (cH, cV, cD)
                sharpened_channel = pywt.waverec2(coeffs, 'haar')
                
                # Asegurarse de que los valores estén dentro del rango permitido
                sharpened_channel = np.clip(sharpened_channel, 0, 255)
                final_sharpened_arrays.append(sharpened_channel)

            # Combinar los canales de color en una sola imagen
            sharpened_img_array = np.stack(final_sharpened_arrays, axis=-1)

            # Convertir de vuelta a una imagen PIL y devolver
            sharpened_image = Image.fromarray(sharpened_img_array.astype(np.uint8))

            return sharpened_image

                
        def estimate_local_sharpness(image, radius=2):
            """
            Estima la nitidez local de la imagen.

            :param image: Imagen PIL a procesar.
            :param radius: Radio para el cálculo de la nitidez local.
            :return: Array de NumPy representando la nitidez local.
            """
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
            sharpness_estimate = ImageChops.subtract(image, blurred_image)
            return np.array(sharpness_estimate)

        def apply_adaptive_sharpen(image, intensity=1.0):
            """
            Aplica un enfoque adaptativo a la imagen.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad general del enfoque.
            :return: Imagen PIL enfocada.
            """
            
            # Asegúrate de que la imagen sea en RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            sharpness_map = estimate_local_sharpness(image)
            # sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150 * intensity))
            # tenemos que ver cómo hacer esto.... solo admite integers, y el 1 es demasiado alto.
            sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=(intensity*100) * 2))

            # Convertir imágenes a arrays para combinarlas
            original_array = np.array(image)
            sharpened_array = np.array(sharpened_image)
            
            #sharpness_map = sharpness_map / sharpness_map.max()  # Normalizar
            
            # Normalizar sharpness_map
            sharpness_map = sharpness_map / sharpness_map.max()

            # Verificar que sharpness_map solo tiene dos dimensiones
            if len(sharpness_map.shape) == 2:
                sharpness_map = sharpness_map[..., np.newaxis]  # Añadir una tercera dimensión
            #print("Sharpness map shape after adding new axis:", sharpness_map.shape)
  

            # Aplicar el enfoque de manera adaptativa
            RMADA_adaptive_sharpened_array = original_array + (sharpened_array - original_array) * sharpness_map

            # Asegurarse de que los valores estén en el rango adecuado
            RMADA_adaptive_sharpened_array = np.clip(RMADA_adaptive_sharpened_array, 0, 255).astype(np.uint8)

            return Image.fromarray(RMADA_adaptive_sharpened_array)
        
        def adjust_RMADA_contrast(image, RMADA_contrast_factor=1.0):
            """
            Ajusta el RMADA_contraste de una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_contrast_factor: Factor por el que se ajustará el RMADA_contraste.
            :return: Imagen PIL con el RMADA_contraste ajustado.
            """
            
            RMADA_contrast_factor = RMADA_contrast_factor * 0.5 + 1
            
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(RMADA_contrast_factor)
        
        def adjust_RMADA_brightness(image, RMADA_brightness_factor=1.0):
            """
            Ajusta el brillo de una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_brightness_factor: Factor por el que se ajustará el brillo.
            :return: Imagen PIL con el brillo ajustado.
            """
            
            RMADA_brightness_factor = RMADA_brightness_factor * 0.5 + 1
            
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(RMADA_brightness_factor)
        
        def adjust_RMADA_saturation(image, RMADA_saturation_factor=1.0):
            """
            Ajusta la saturación de una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_saturation_factor: Factor por el que se ajustará la saturación.
            :return: Imagen PIL con la saturación ajustada.
            """
            
            RMADA_saturation_factor = RMADA_saturation_factor * 0.5 + 1
            
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(RMADA_saturation_factor)

        def add_film_RMADA_noise(image, RMADA_noise_intensity=0.5):
            """
            Añade ruido de película a una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_noise_intensity: Intensidad del ruido añadido.
            :return: Imagen PIL con ruido de película.
            """
            # Convertir la imagen a un array de NumPy
            img_array = np.array(image)

            # Generar ruido
            RMADA_noise = np.random.normal(loc=0, scale=255 * RMADA_noise_intensity, size=img_array.shape)

            # Añadir ruido a la imagen
            noisy_img_array = img_array + RMADA_noise

            # Asegurarse de que los valores estén dentro del rango permitido
            noisy_img_array = np.clip(noisy_img_array, 0, 255)

            # Convertir de vuelta a una imagen PIL
            noisy_image = Image.fromarray(noisy_img_array.astype(np.uint8))

            return noisy_image
                
        def overlay_vignette(image, strength=0.5):
            """
            Aplica un efecto de viñeta a una imagen, con un control adicional sobre la transparencia.
            :param image: Imagen PIL a procesar.
            :param strength: Determina cuán lejos hacia el centro se extiende la viñeta.
            :param transparency: Controla la opacidad máxima de la viñeta.
            :return: Imagen PIL con efecto de viñeta.
            """
            transparency=0.5
            # Crear la máscara de viñeta
            width, height = image.size
            x, y = np.ogrid[:height, :width]
            # Calcula la distancia al centro y la normaliza
            center = np.array([height / 2, width / 2])
            distance_to_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            max_distance = np.sqrt(center[0]**2 + center[1]**2)
            #normalized_distance = distance_to_center / max_distance
            # Calcula el gradiente de viñeta basado en la intensidad y la transparencia
            #vignette_gradient = 1 - normalized_distance**strength
            #vignette_gradient = vignette_gradient * transparency + (1 - transparency)
            
            # Modificar la función del gradiente para hacer el centro menos oscuro
            normalized_distance = distance_to_center / max_distance
            vignette_gradient = 1 - (normalized_distance ** (strength / 2))
            
            # Aplica la transparencia al gradiente
            vignette_gradient = vignette_gradient * transparency + (1 - transparency)
        
            
            vignette_mask = Image.fromarray((vignette_gradient * 255).astype(np.uint8), 'L')

            # Aplica la máscara de viñeta a la imagen utilizando el modo "multiply"
            vignette_image = Image.new("RGB", image.size)
            vignette_image.paste(image)
            vignette_image.putalpha(vignette_mask)
            result_image = Image.composite(vignette_image, Image.new("RGB", image.size, "black"), vignette_mask)
            return result_image

        

        def add_RMADA_vignette(image, strength=0.5):
            """
            Añade un efecto de viñeta a una imagen.

            :param image: Imagen PIL a procesar.
            :param strength: Intensidad del efecto de viñeta.
            :return: Imagen PIL con efecto de viñeta.
            """
            # Calcular el gradiente de viñeta
            width, height = image.size
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / (width / 2)
            y = (y - height / 2) / (height / 2)
            
            #gradient = 1 - np.sqrt(x**2 + y**2) * strength
            #gradient = np.clip(gradient, 0, 1)
            
            # Modificar el gradiente para que el oscurecimiento sea más suave
            # gradient = np.sqrt(x**2 + y**2)
            # gradient = np.clip(2 - strength * gradient, 0, 1)

            # Ajustar el gradiente y aplicar el factor de intensidad
            #gradient = np.sqrt(x**2 + y**2)
            #gradient = 1 - np.clip(gradient * strength * 2, 0, 1)
            
            # Modificar el gradiente para un incremento gradual desde el centro hacia los bordes
            gradient = np.sqrt(x**2 + y**2)
            gradient = np.clip(2- gradient ** (strength * 10), 0, 1)


            # Invertir el gradiente para oscurecer los bordes
            # gradient = 1 - gradient
            
            # Aplicar el gradiente a la imagen
            vignette = Image.fromarray((gradient * 255).astype(np.uint8), 'L')
            vignette = vignette.resize(image.size)

            # Combinar la imagen original con la viñeta
            return Image.composite(image, Image.new("RGB", image.size, "black"), vignette)

            # Aplicar el gradiente a la imagen
            # RMADA_vignette = Image.fromarray((gradient * 255).astype(np.uint8), 'L')
            # RMADA_vignette = RMADA_vignette.resize(image.size)
            # RMADA_vignette_image = ImageOps.colorize(RMADA_vignette, (0, 0, 0), (255, 255, 255))

            # Combinar la imagen original con la viñeta
            # return Image.composite(image, RMADA_vignette_image, RMADA_vignette)
        
        if self.config.RMADA_enable:
            print("Applying rMada ProImage")

            for i in range(len(processed.images)):
                try:
                    pil_image = Image.fromarray(np.array(processed.images[i]))
                except Exception as e:
                    print(f"Error al convertir la imagen: {e}")
                    continue
                
                pil_output = pil_image
                
                if self.config.RMADA_sharpenweight > 0:
                    # print(f"LOG RMADA_sharpenweight: {self.config.RMADA_sharpenweight}")
                    sharpened_image = apply_custom_sharpen(pil_output, self.config.RMADA_sharpenweight)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_edge_detection_sharpening > 0:
                    # print(f"LOG RMADA_edge_detection_sharpening: {self.config.RMADA_edge_detection_sharpening}")
                    sharpened_image = apply_edge_detection_sharpen(pil_output, self.config.RMADA_edge_detection_sharpening)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_wavelet_sharpening > 0:
                    # print(f"LOG RMADA_wavelet_sharpening: {self.config.RMADA_wavelet_sharpening}")
                    sharpened_image = apply_wavelet_sharpen(pil_output, self.config.RMADA_wavelet_sharpening)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_adaptive_sharpened > 0:
                    # print(f"LOG RMADA_adaptive_sharpened: {self.config.RMADA_adaptive_sharpened}")
                    sharpened_image = apply_adaptive_sharpen(pil_output, self.config.RMADA_adaptive_sharpened)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_contrast != 0:
                    # print(f"LOG RMADA_contrast: {self.config.RMADA_contrast}")
                    sharpened_image = adjust_RMADA_contrast(pil_output, self.config.RMADA_contrast)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_brightness != 0:
                    # print(f"LOG RMADA_brightness: {self.config.RMADA_brightness}")
                    sharpened_image = adjust_RMADA_brightness(pil_output, self.config.RMADA_brightness)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)
                
                if self.config.RMADA_saturation != 0:
                    # print(f"LOG RMADA_saturation: {self.config.RMADA_saturation}")
                    sharpened_image = adjust_RMADA_saturation(pil_output, self.config.RMADA_saturation)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_noise != 0:
                    # print(f"LOG RMADA_noise: {self.config.RMADA_noise}")
                    sharpened_image = add_film_RMADA_noise(pil_output, self.config.RMADA_noise)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_vignette != 0:
                    # print(f"LOG RMADA_vignette: {self.config.RMADA_vignette}")
                    sharpened_image = add_RMADA_vignette(pil_output, self.config.RMADA_vignette)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_vignette_overlay != 0:
                    # print(f"LOG RMADA_vignette_overlay: {self.config.RMADA_vignette_overlay}")
                    sharpened_image = overlay_vignette(pil_output, self.config.RMADA_vignette_overlay)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                processed.images[i] = np.array(pil_output)
                
        OmegaConf.save(self.config, CONFIG_PATH)
        
    def process_batch(self, p, *args, **kwargs):
        self.step_limit = 0
