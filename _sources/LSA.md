# LSA (Latent Semantic Analysis)

Latent Semantic Analysis (LSA) merupakan sebuah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi. Secara umum, langkah-langkah LSA dalam penilaian esai adalah sebagai berikut:

## 1. Text Preprocessing

Preprocessing adalah proses normalisasi teks sehingga informasi yang dimuat merupakan bagian yang padat dan ringkas namun tetap merepresentasikan informasi yang termuat didalamnya. Dalam tahap ini, terdapat beberapa proses diantaranya:

- Stopwords Removal : Pada stopwords removal, kata-kata yang tergolong sebagai kata depan, kata penghubung, dan kata-kata lain yang tidak mewakili makna dari kalimat akan dieliminasi.

- Stemming : Langkah berikutnya adalah stemming. Pada proses ini, kata akan dinormalkan menjadi kata dasar pembentuk kata tersebut. Caranya adalah dengan menghilangkan imbuhan yang melekat pada kata, sehingga hasilnya adalah kata dasarnya. Apabila dalam Bahasa Inggris, proses stemming bisa mengikutsertakan pengembalian bentuk tense dari kata kerja bentuk ke-2 atau ke-3 menjadi kata kerja bentuk ke-1.

## 2. Term-document Matrix

Setelah melalui stopwords removal dan stemming, matriks term-document dibangun dengan menempatkan kata hasil proses stemming (term) ke dalam baris. Matriks ini disebut term-document matrix. Setiap baris mewakili sebuah kata yang unik, sedangkan setiap kolom mewakili konteks dari mana kata-kata tersebut diambil. Konteks yang dimaksud bisa berupa kalimat, paragraf, atau seluruh bagian dari teks.
Di bawah ini merupakan contoh term-document matrix:

 <img src="pw1.jpg" width=1000 height=400  style="margin-left:auto; margin-right:auto; display: block;" />

 Pada tabel di atas, baris pertama mewakili stemmed term (term 1, term 2, dst), dan bagian kolom mewakili konteks, yaitu teks. Nilai yang terletak pada setiap cell pada tabel menunjukkan berapa kali sebuah term muncul dalam sebuah dokumen. Contohnya, term 1 muncul 1 kali pada dokumen ke-1, dan muncul 2 kali pada dokumen ke-2, namun term 1 tidak muncul pada dokumen 3, dan seterusnya.

 ## 3. Singular Value Decomposition

 Singular Value Decomposition (SVD) adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan term-document matrix. SVD merupakan teorema aljabar linier yang menyebutkan bahwa persegi panjang dari term-document matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu :
 
– Matriks ortogonal U

– Matriks diagonal S

– Transpose dari matriks ortogonal V

Yang dirumuskan dengan :

<img src="pw2.jpg"  style="margin-left:auto; margin-right:auto; display: block;" />

Hasil dari proses SVD adalah vektor yang akan digunakan untuk menghitung similaritasnya dengan pendekatan cosine similarity.

## 4. Cosine Similarity Measurement

Cosine similarity digunakan untuk menghitung nilai kosinus sudut antara vektor dokumen dengan vektor kueri. Semakin kecil sudut yang dihasilkan, maka tingkat kemiripan esai semakin tinggi.
Formula dari cosine similarity adalah sebagai berikut:

<img src="pw3.jpg"  style="margin-left:auto; margin-right:auto; display: block;" />

Dari hasil cosine similarity, akan didapatkan nilai yang akan dibandingkan dengan penilaian manusia untuk diuji selisih nilainya.

