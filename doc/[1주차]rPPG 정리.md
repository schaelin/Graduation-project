## [1주차]rPPG 정리  

> rPPG 측정방법, 측정 알고리즘에 대하여 설명   
> 링크 걸린 논문 이해하면 좋아요(특히 CHROM, ICA, 두번째랑 세번째 링크)
   
    
### 1. PPG란?

---

PPG(Photoplethysmography)는 심장박동과 동기된 혈량의 변화를 측정한 것을 의미한다.

- 측정 원리  

    1) 심장박동에 의하여 생성된 압력에 의해 혈관내에서 혈액의 흐름이 생긴다. 심장박동이 발생할 때마다 압력은 신체의 말단 모세혈관까지 작용하며, 손가락 끝의 혈관까지도 압력이 작용한다.(그림)  
    
    2) 손가락 끝 모세혈관의 동맥 혈액은 세포조직으로 혈액을 공급하고, 정맥으로 들어가서 다시 심장으로 되돌아간다.(a)  
    
    3) 심장박동에 동기되어 손가락 끝 혈관에서의 동맥혈량(Arterial Blood volume)이 증가하고 줄어드는 상태가 반복된다.(b)  

    <center><img src="https://mblogthumb-phinf.pstatic.net/20130117_33/nbhbae_13583934996017kQKH_JPEG/PPG_%B8%C6%C6%C4_%C3%F8%C1%A4_%BF%F8%B8%AE1.jpg?type=w2"></center>

    그림 출처: [https://m.blog.naver.com/nbhbae/50159924861](https://m.blog.naver.com/nbhbae/50159924861) 

    <br>
### 2. RPPG란?

---

RPPG(Remote-Photoplethysmography)는 비접촉식으로 심장박동과 동기된 혈량의 변화를 측정하는 것을 의미한다.

- 장점

    기존의 PPG 측정은 피험자의 신체에 기구를 장착하여 신호를 추출하기 때문에 사용하는 과정에서 불편함 발생 

    RPPG는 기구를 사용하지 않고 접촉없이 피부에 반사되는 빛을 이용하여 혈량의 변화를 측정할 수 있다.

- 측정 원리

    (1) 심장박동에 의하여 생성된 압력에 의해 혈관내에서 혈액의 흐름이 생기며, 혈량 변화에 의해 피부의 색에 미세한 변화가 발생한다.

    (2) 적외선 또는 근적외선 광원을 이용하여 피부에서 반사되는 빛의 변화를 통해 혈량이 증가하고 줄어드는 상태가 반복되는 것을 포착한다.
    
     
     <br>
### 3. RPPG 측정 과정

---

	(1) 동영상 프레임으로부터 얼굴 영역(ROI) 검출
	
	(2) 추출된 ROI 영역으로부터 시계열 신호 추출
	  - 혈류 흐름에 따른 피부색의 미세변화는 주기성을 가지는 시계열 신호적 특징을 가진다.
	  
	(3) 다양한 알고리즘을 적용하여 시계열 신호(raw signal)에서 혈류 변화 관련 신호 추출

<br>
    
### 4. RPPG 측정 알고리즘

---

혈량 변화를 측정하는 방법에 따라 rppg의 알고리즘과 성능이 달라진다.

[Wang(2016), Algorithmic Principles of Remote-PPG](https://ieeexplore.ieee.org/abstract/document/7565547) 에서는 혈량 변화 측정을 위해 피부에서 반사되는 빛을 처리하는 여러가지 알고리즘에 대하여 설명

> 어떤 알고리즘이 있는지 종류정도만 파악하면 될 것 같다.   
> ICA, CHROM 관련 논문은 읽어보는 것이 좋을 듯   

   
**1) BSS-based methods(PCA)** 

   BSS(Blind Source Separation) Techniques는 신호에 대한 정보 없이 혼합 신호 집합에서 소스 신호 집합을 분리하는 기술을 의미한다.

   _[Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381)_

   BSS 방법에는 PCA(Principal Component Analysis) 기반 알고리즘과 ICA(Independent Component Analysis)기반 알고리즘이 있다.

         PCA : 데이터를 가장 잘 설명하는 축을 찾는 방법
         ICA : 독립성이 최대가 되는 벡터를 찾는 방법

<img src="https://ifh.cc/g/4pLYQ4.jpg" width="400" height="360">

   그림은 ICA로 PPG 신호를 추출하는 과정이다.

    [검출과정]
    a) 비디오 프레임에서 얼굴 피부 영역을 검출
    b) 검출한 ROI 영역에서 Red, Green, Blue 컬러채널을 분리
    c) 각 컬러채널에서 신호를 추출
    d) 추출한 신호에 ICA를 적용하여 각 채널별로 분리된 소스 신호를 검출

   _M.-Z. Poh, D. McDuff, and R. Picard, “Advancements in noncontact, multiparameter physiological measurements using a webcam, ” Biomedical Engineering, IEEE Trans. on, vol. 58, no. 1, pp. 7-11, Jan. 2011._

      
**2) Model-based Methods(PBV/CHROM)**

   model-based methods는 다른 구성 요소의 color vector를 이용하여 디믹싱을 진행하여 average skin reflection(DC Level)에 대한 의존성을 제거한 알고리즘이다.

   model-based methods에는 PBV(normalized volume pulse vector)와 CHROM(chrominance)방법이 있다.

      
**2-1) PBV**

	the motion robustness improved method using the blood volume pulse signature

	define PBV as the relative PPG-amplitude in the normalized RGB-color channels of the video camera registering a stationary skin-region

_G. de Haan and A. van Leest, “Improved motion robustness of remotePPG by using the blood volume pulse signature, ” Physiological Measurement, vol. 35, no. 9, pp. 1913-1922, Oct. 2014._
      
      
**2-2) CHROM(★)**

	the motion robust method based on the standardized skin-tone assumption

	color difference를 통해 specular reflection component 제거

_[G. de Haan and V. Jeanne,“Robust pulse rate from chrominance-based rPPG, ” Biomedical Engineering, IEEE Trans. on, vol. 60, no. 10, pp. 2878-2886, Oct. 2013.](https://ieeexplore.ieee.org/abstract/document/6523142)_
      
      
**3) Data-driven method(2SR)**

---

	2SR(Spartial Subspace Rotation) : robustness of remote photoplenthysmography

	a recent method exploiting the skin-pixel distribution in the image domain

dependent skin color space 를 생성하고 통계적 분포를 기반으로 결정된 색조변화를 추적하여 신호 측정

_W. Wang, S. Stuijk, and G. de Haan,“A novel algorithm for remote photoplethysmography: Spatial subspace rotation, ” Biomedical Engineering, IEEE Transactions on, vol. PP, no. 99, pp. 1-1, 2015._
       
       
**4) POS Algorithm**

---

POS(plane-othogonal-to-skin)는 펄스 추출을 위해 일시적으로 정규화된 RGB 공간에서 피부 톤과 직교하는 평면을 정의
