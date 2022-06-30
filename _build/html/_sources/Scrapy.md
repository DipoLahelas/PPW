# Melakukan Crawling dengan Scrapy

## Membuat kode Scrapy

Disini saya membuat kode untuk melakukan crawling pada link "Baca selengkapya" yang ada di pta.trunojoyo.ac.id.
Saya mengambil data dari 10 halaman awal yang ada di pta.trunojoyo.
kemudian dimasukkan  ke file csv.

```python
from numpy import append
import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/10/1']
        for i in range (2,11):
            tambah = 'https://pta.trunojoyo.ac.id/c_search/byprod/10/'+ str(i)
            start_urls.append(tambah)

        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1, 6):
            yield {
                'link':response.css('#content_journal > ul > li:nth-child(' +str(i)+ ') > div:nth-child(3) > a::attr(href)').extract()
            }
```

Kemudian masukan perintah dibawah ini didalam terminal untuk melakukan crawling data link di pta.trunojoyo dan memasukkan data ke dalam file csv:

```
Scrapy runspider <nama-py-scrapy>.py -O <nama-file>.csv
```

Hasil Crawling link jurnal : [file](https://github.com/egi-190137/topic-modelling-sklearn/blob/main/contents/link.csv)
|link                                                   |
|-------------------------------------------------------|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007|
|https://pta.trunojoyo.ac.id/welcome/detail/040411100468|
|https://pta.trunojoyo.ac.id/welcome/detail/040411100468|
|https://pta.trunojoyo.ac.id/welcome/detail/040411100468|
|https://pta.trunojoyo.ac.id/welcome/detail/040411100468|
|https://pta.trunojoyo.ac.id/welcome/detail/040411100468|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100120|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100120|

## Crawling Judul dan abstraksi Jurnal

Kemudian buat kode Scrapy kedua untuk melakukan ekstraksi data dari link - link yang sudah diambil tadi

```python
import scrapy
import pandas as pd



class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        
        dataCSV = pd.read_csv('link.csv')
        indexData = dataCSV.iloc[:, [0]].values
        arrayData = []
        for i in indexData:
            ambil = i[0]
            arrayData.append(ambil)
        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield {
            'judul_TA': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'pembuat': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(2) > span::text').extract(),
            'Pembimbing_1': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(3) > span::text').extract(),
            'Pembibing_2': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(4) > span::text').extract(),
            'abstrak': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract()    
        }
```

Lalu masukan perintah dibawah ini didalam terminal untuk melakukan crawling data PTA yang berasal dari link yang sudah di ekstraksi tadi dan memasukkan data ke dalam file csv:

```
Scrapy runspider <nama-py-scrapy>.py -O <nama-file>.csv
```

Hasil Crawling jududl dan abstraksi jurnal: [file](https://github.com/egi-190137/topic-modelling-sklearn/blob/main/contents/detail_pta.csv)

| Judul                                   | Abstraksi                                                         |
| --------------------------------------- | ----------------------------------------------------------------- |
| Gerak Pekerja Pada Game Real Time S ... | Gerak pekerja ada pada game yang memiliki genre RTS (Real-Tim ... |
| PEMANFAATAN TOGAF ADM UNTUK PERANCA ... | Penyusunan Sistem Informasi Dinas Perindustrian & Perdagangan ... |
| RANCANG BANGUN MANAJEMEN PEMBELAJAR ... | Penggunaan teknologi mobile saat ini sangat marak, disamping  ... |
| SISTEM PENDUKUNG KEPUTUSAN PEMILI ...   | Sumber daya manusia mutlak dibutuhkan untuk kemajuan suatu per... |
| SISTEM PENENTUAN STATUS GIZI PASI ...   | Di Indonesia masalah perkembangan gizi adalah masalah yang per... |
