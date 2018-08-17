from icrawler.builtin import GoogleImageCrawler

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

attractionList = ['tower eiffel', 'statue of liberty', 'niagara falls', 'colosseum', 'pyramid']
attractionFolderList = ['eiffel', 'liberty', 'niagara', 'colosseum', 'pyramid']

for idx, val in enumerate(attractionList):
    google_crawler = GoogleImageCrawler(
        feeder_threads=10,
        parser_threads=10,
        downloader_threads=10,
        storage={'root_dir': 'data/'+attractionFolderList[idx]})
    google_crawler.session.verify = False
    filters = dict(type='photo') #사진만
    # 키워드로 돌면서 1000장 크롤링
    google_crawler.crawl(keyword=val, filters=filters, max_num=1000, file_idx_offset=0)
