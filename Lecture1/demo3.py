import requests
from bs4 import BeautifulSoup
import os


def download_all_images(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        image_elements = soup.find_all('img')
        if not os.path.exists('all_images'):
            os.makedirs('all_images')

        for index, img in enumerate(image_elements):
            img_url = img.get('src')
            if img_url:
                try:
                    img_response = requests.get(img_url, headers=headers)
                    if img_response.status_code == 200:
                        file_extension = os.path.splitext(img_url)[1]
                        if not file_extension:
                            file_extension = '.jpg'
                        file_path = os.path.join('all_images', f'image_{index + 1}{file_extension}')
                        with open(file_path, 'wb') as f:
                            f.write(img_response.content)
                        print(f'已下载图片: {file_path}')
                    else:
                        print(f'下载图片失败，状态码: {img_response.status_code}')
                except requests.RequestException as img_err:
                    print(f'下载图片时请求出错: {img_err}')
            else:
                print('未获取到图片链接')
    except requests.RequestException as e:
        print(f'请求出现错误: {e}')


if __name__ == '__main__':
    target_url = 'https://sh.fang.lianjia.com/loupan/p_hfglbdbnocw/'
    download_all_images(target_url)
