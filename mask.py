import json
import numpy as np
import cv2
import os

# 클래스와 해당 색상 정의 (RGB 형식으로 수정)
CLASS_TO_COLOR = {
    "Scratched": (255, 0, 0),  # 빨강 (Red)
    "Crushed": (255, 0, 0),    
    "Breakage": (255, 0, 0),  
    "Separated": (255, 0, 0) 
}

def create_masks_from_json(image_dir, json_dir, output_dir, log_file):
    """
    Create mask images from annotation JSON files for each image.

    :param image_dir: Directory containing the train images.
    :param json_dir: Directory containing the annotation JSON files.
    :param output_dir: Directory to save generated mask images.
    :param log_file: Path to the log file for recording errors.
    """
    # 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 로그 파일 초기화
    with open(log_file, "w") as log:
        log.write("Error Log for Mask Generation:\n\n")

    # 이미지 디렉토리의 모든 이미지 파일 가져오기
    image_files = sorted(os.listdir(image_dir))

    for image_file in image_files:
        try:
            # 이미지 경로와 연결된 JSON 파일 확인
            json_file = os.path.join(json_dir, os.path.splitext(image_file)[0] + ".json")
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"JSON file not found for {image_file}. Skipping...")

            # 이미지 로드 (이미지를 BGR 형식으로 변환)
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            
            # 이미지를 BGR에서 RGB로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image_rgb.shape

            # JSON 파일 로드
            with open(json_file, "r") as f:
                data = json.load(f)

            # 빈 마스크 초기화
            mask = np.zeros((height, width, 3), dtype=np.uint8)

            # 어노테이션 처리
            annotations = data.get("annotations", [])
            for ann in annotations:
                segmentation = ann["segmentation"]
                damage_type = ann["damage"]

                # segmentation 좌표를 폴리곤으로 변환하여 채우기
                for segment in segmentation:
                    # segment가 리스트 안에 또 다른 리스트가 있을 수 있음
                    # 중첩된 리스트를 처리하도록 수정
                    if isinstance(segment[0], list):
                        for sub_segment in segment:
                            points = np.array(sub_segment, dtype=np.int32)
                            color = CLASS_TO_COLOR.get(damage_type, (0, 0, 0))  # 정의되지 않은 클래스는 검정색
                            cv2.fillPoly(mask, [points], color)
                    else:
                        points = np.array(segment, dtype=np.int32)
                        color = CLASS_TO_COLOR.get(damage_type, (0, 0, 0))  # 정의되지 않은 클래스는 검정색
                        cv2.fillPoly(mask, [points], color)

            # 마스크 저장 (파일명을 _mask.png로 설정)
            mask_file_name = os.path.splitext(image_file)[0] + "_mask.png"
            mask_output_path = os.path.join(output_dir, mask_file_name)

            # RGB를 BGR로 다시 변환하여 저장 (OpenCV는 기본적으로 BGR을 사용)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(mask_output_path, mask_bgr)

            print(f"Saved mask for {image_file} to {mask_output_path}")

        except Exception as e:
            # 오류 발생 시 로그 기록
            with open(log_file, "a") as log:
                log.write(f"Error processing {image_file}: {str(e)}\n")
            print(f"Error processing {image_file}: {str(e)}. Skipping...")

def process_all_datasets(base_dir, datasets):
    """
    Process the 'train', 'test', 'valid' datasets for mask generation.

    :param base_dir: Base directory containing all datasets.
    :param datasets: List of dataset directories ('train', 'test', 'valid').
    """
    for dataset in datasets:
        image_dir = os.path.join(base_dir, dataset, 'image')
        json_dir = os.path.join(base_dir, dataset, 'label')
        output_dir = os.path.join(base_dir, dataset, 'mask')
        log_file = os.path.join(base_dir, dataset, 'error_log.txt')
        
        print(f"Processing {dataset} dataset...")
        create_masks_from_json(image_dir, json_dir, output_dir, log_file)

# 실행
base_dir = "/WD/심신개/car_damage_data_v"  # 데이터셋이 위치한 기본 경로
datasets = ['train', 'test', 'valid']  # 처리할 데이터셋 목록

process_all_datasets(base_dir, datasets)