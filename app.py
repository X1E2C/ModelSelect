import json
import readchar
import sys
from difflib import get_close_matches, SequenceMatcher
from rich.console import Console
from rich.syntax import Syntax
from huggingface_hub import HfApi
import os
import subprocess

def select_model_with_arrows_or_index(models, page_size=50):
    selected_index = 0
    current_page = 0
    total_pages = (len(models) + page_size - 1) // page_size
    input_buffer = ""

    def render_list():
        sys.stdout.write("\033c" if sys.platform != "win32" else "cls")
        start_index = current_page * page_size
        end_index = min(start_index + page_size, len(models))
        print(f"\nModeller (Sayfa {current_page + 1}/{total_pages}):\n")
        for i in range(start_index, end_index):
            model = models[i]
            if i == selected_index:
                print(f"> {i}: {model['modelId']}  ✅")
            else:
                print(f"  {i}: {model['modelId']}")
        print("\n⬆ Yukarı / ⬇ Aşağı tuşları ile gezin. Enter ile model seçin veya model index numarasını girin:")

    render_list()
    print("\nGiriş: ", end="", flush=True)

    while True:
        key = readchar.readkey()
        if key.isdigit():
            input_buffer += key
            print(f"{input_buffer}", end="\rGiriş: ", flush=True)
        elif key == readchar.key.ENTER:
            if input_buffer:
                try:
                    index = int(input_buffer)
                    if 0 <= index < len(models):
                        return index
                    else:
                        print("\n⚠️ Geçersiz index girdiniz.")
                except ValueError:
                    print("\n⚠️ Geçersiz giriş.")
                finally:
                    input_buffer = ""
                    print("\nGiriş: ", end="", flush=True)
            else:
                current_page = (current_page + 1) % total_pages
                render_list()
                print("\nGiriş: ", end="", flush=True)
        elif key == readchar.key.BACKSPACE:
            input_buffer = input_buffer[:-1]
            print(f"{input_buffer} ", end="\rGiriş: ", flush=True)
        elif key == readchar.key.UP:
            if selected_index > 0:
                selected_index -= 1
            elif selected_index == 0 and current_page > 0:
                current_page -= 1
                selected_index = (current_page + 1) * page_size - 1
            render_list()
            print(f"\n✅ Seçili Model: {selected_index}: {models[selected_index]['modelId']}")
            print("\nGiriş: ", end="", flush=True)
        elif key == readchar.key.DOWN:
            if selected_index < len(models) - 1:
                selected_index += 1
            elif selected_index == (current_page + 1) * page_size - 1 and current_page < total_pages - 1:
                current_page += 1
                selected_index = current_page * page_size
            render_list()
            print(f"\n✅ Seçili Model: {selected_index}: {models[selected_index]['modelId']}")
            print("\nGiriş: ", end="", flush=True)

def check_model_gguf(model_info):
    files = model_info.siblings
    gguf_files = [f for f in files if f.rfilename.endswith(".gguf")]
    return gguf_files

def display_model_info(model_info):
    model_json = {
        "modelId": model_info.modelId,
        "sha": model_info.sha,
        "lastModified": str(model_info.lastModified),
        "private": model_info.private,
        "tags": model_info.tags,
        "downloads": model_info.downloads,
        "likes": model_info.likes,
    }
    console = Console()
    syntax = Syntax(json.dumps(model_json, indent=4, ensure_ascii=False), "json", theme="monokai", line_numbers=True)
    console.print("\n🔹 **Seçilen Modelin JSON Bilgileri:**", style="bold cyan")
    console.print(syntax)

import time

def download_model_to_path(model_id, save_path, gguf_file=None, max_retries=3, timeout=600):
    try:
        os.makedirs(save_path, exist_ok=True)  # GGUF klasörünün var olup olmadığını kontrol et
        if gguf_file:
            print(f"\nGGUF dosyaları arasından seçim yapın:")
            for i, gfile in enumerate(gguf_file):
                print(f"{i}: {gfile.rfilename}")
            
            while True:
                try:
                    gguf_index = int(input("\nHangi GGUF dosyasını indirmek istersiniz? Index giriniz: "))
                    if 0 <= gguf_index < len(gguf_file):
                        selected_gguf = gguf_file[gguf_index]
                        print(f"\n{model_id} modelinin {selected_gguf.rfilename} dosyası indiriliyor...")
                        command = f'huggingface-cli download {model_id} --local-dir "{save_path}" --filename "{selected_gguf.rfilename}"'
                        subprocess.run(command, shell=True, check=True)
                        print("✅ GGUF dosyası başarıyla indirildi.")
                        return
                    else:
                        print("⚠️ Geçersiz index seçimi.")
                except ValueError:
                    print("⚠️ Geçersiz giriş.")
        else:
            print(f"\n{model_id} modeli indiriliyor...")

            command = f'huggingface-cli download {model_id} --local-dir "{save_path}"'
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
                    if result.returncode == 0:
                        print("\n✅ Model başarıyla indirildi.")
                        break  # Başarılı olursa döngüden çık
                    else:
                        print(f"⚠️ Model indirilirken hata oluştu (Deneme {retry_count+1}/{max_retries}):\n{result.stderr}")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Model indirme işlemi zaman aşımına uğradı (Deneme {retry_count+1}/{max_retries}). Bağlantıyı kontrol edin ve tekrar deneniyor...")
                
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(10)  # 10 saniye bekleyerek tekrar dene
                else:
                    print("⛔ Maksimum deneme sayısına ulaşıldı. Model indirilemedi.")
                    return

            print("\nModel GGUF formatına dönüştürülecek.")
            
            convert_success = False
            while not convert_success:
                print("\nQuantization seçenekleri:")
                print("1: Q4_K_M (4-bit quantization, medium quality)")
                print("2: Q5_K_M (5-bit quantization, higher quality)")
                print("3: Q8_0 (8-bit quantization, highest quality)")
                
                while True:
                    choice = input("\nLütfen quantization seçeneğini girin (1-3): ").strip()
                    if choice in ['1', '2', '3']:
                        quant_type = {
                            '1': 'q4_k_m',
                            '2': 'q5_k_m',
                            '3': 'q8_0'
                        }[choice]
                        break
                    print("⚠️ Geçersiz seçim. Lütfen 1-3 arası bir sayı girin.")
                
                output_path = os.path.join(save_path, "model_converted.gguf")
                convert_script_path = "C:\\Users\\ekink\\OneDrive\\Masaüstü\\HubAPI\\llama.cpp\\convert_hf_to_gguf.py"
                convert_command = f'python "{convert_script_path}" "{save_path}" --outfile "{output_path}" --outtype {quant_type}'

                retry_count = 0
                while retry_count < max_retries:
                    try:
                        result = subprocess.run(convert_command, shell=True, capture_output=True, text=True, timeout=timeout)
                        if result.returncode == 0:
                            print(f"✅ Model başarıyla GGUF formatına dönüştürüldü: {output_path}")
                            convert_success = True
                            break
                        else:
                            print(f"⚠️ GGUF formatına dönüştürme sırasında hata oluştu (Deneme {retry_count+1}/{max_retries}):\n{result.stderr}")
                    except subprocess.TimeoutExpired:
                        print(f"⚠️ GGUF formatına dönüştürme işlemi zaman aşımına uğradı (Deneme {retry_count+1}/{max_retries}). Tekrar deneniyor...")
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(10)  # 10 saniye bekleyerek tekrar dene
                    else:
                        print("⛔ Maksimum deneme sayısına ulaşıldı. Farklı bir quantization seçeneği deneyin.")
                        break
                
                if convert_success:
                    break
                
                retry = input("\nYeniden quantization seçmek ister misiniz? (E/H): ").strip().lower()
                if retry != 'e':
                    print("⛔ GGUF dönüştürme işlemi iptal edildi.")
                    return
        
        print(f"✅ İşlem başarıyla tamamlandı.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ İşlem sırasında hata oluştu: {e}")
    except Exception as e:
        print(f"⚠️ Beklenmeyen bir hata oluştu: {e}")

def find_closest_model_name(search_substring, api):
    all_models = api.list_models(limit=5000)
    candidate_names = [m.modelId for m in all_models]
    matches = get_close_matches(search_substring, candidate_names, n=3, cutoff=0.5)
    if matches:
        return matches[0]
    best_match = None
    best_ratio = 0.0
    for candidate in candidate_names:
        ratio = SequenceMatcher(None, search_substring.lower(), candidate.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate
    return best_match if best_ratio > 0.4 else None

def main():
    api = HfApi()
    console = Console()
    
    print("Hugging Face Model Arama")
    search_substring = input("Lütfen aramak istediğiniz model adını veya ID'sini giriniz: ").strip()
    
    print("\nModeller aranıyor...\n")
    models = api.list_models(search=search_substring)
    models = sorted(models, key=lambda x: x.modelId.lower())
    
    if not models:
        print("Aradığınız model bulunamadı. Yakın eşleşmeler aranıyor...\n")
        closest_match = find_closest_model_name(search_substring, api)
        if closest_match:
            print(f"\n📌 En yakın eşleşme: {closest_match}")
            cevap = input(f"Bunu mu demek istediniz: '{closest_match}'? (E/H): ").strip().lower()
            if cevap == 'e':
                models = api.list_models(search=closest_match)
                models = sorted(models, key=lambda x: x.modelId.lower())
                if not models:
                    print("⚠️ Önerilen model bulunamadı.")
                    return
            else:
                print("⚠️ Arama kriterlerine uyan model bulunamadı.")
                return
        else:
            print("⚠️ Hiçbir benzer model bulunamadı.")
            return
    
    model_list = [{"index": i, "modelId": model_info.modelId} for i, model_info in enumerate(models)]
    
    print("\nJSON Formatında Model Listesi:")
    print(json.dumps({"modeller": model_list}, indent=4))
    
    selected_index = select_model_with_arrows_or_index(model_list, page_size=50)
    selected_model = models[selected_index]
    print(f"\n✅ Seçilen model: {selected_model.modelId}")
    
    selected_model_info = api.model_info(selected_model.modelId)
    gguf_files = check_model_gguf(selected_model_info)
    
    if gguf_files:
        print("\n✅ Bu model GGUF formatında dosyalara sahiptir.")
    else:
        print("\n⚠️ Bu modelde GGUF formatında dosya bulunmamaktadır. İndirme sonrası dönüştürme işlemi yapılacaktır.")
    
    display_model_info(selected_model_info)
    
    download_choice = input("\nBu modeli indirmek istiyor musunuz? (E/H): ").strip().lower()
    if download_choice == 'e':
        save_path = input("Modelin indirileceği dizini belirtiniz (örn: ./indirilen_modeller): ").strip()
        os.makedirs(save_path, exist_ok=True)
        download_model_to_path(selected_model.modelId, save_path, gguf_files if gguf_files else None)
    
    print("\n✅ Program sonlandı.")

if __name__ == "__main__":
    main()
