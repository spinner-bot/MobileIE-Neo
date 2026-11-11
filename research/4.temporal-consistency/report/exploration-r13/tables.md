# Earlier training context controls

Current task supervision/optimizer/inference unchanged; spatial full1000 replay passed; fixed final1000/two seeds.

## test

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_context | causal | False | half01 | 19.12484 | 0.83053 | 2.31066 | 1.96422 |
| concat_aligned_context | causal | False | raw | 19.11910 | 0.83076 | 2.44605 | 1.95751 |
| concat_aligned_context | causal | True | half01 | 19.12484 | 0.83053 | 2.31066 | 1.96422 |
| concat_aligned_context | causal | True | raw | 19.11910 | 0.83076 | 2.44605 | 1.95751 |
| concat_aligned_context | current | False | half01 | 19.12535 | 0.83053 | 2.31888 | 1.96423 |
| concat_aligned_context | current | False | raw | 19.11948 | 0.83072 | 2.45719 | 1.95793 |
| concat_aligned_context | current | True | half01 | 19.12535 | 0.83053 | 2.31888 | 1.96423 |
| concat_aligned_context | current | True | raw | 19.11948 | 0.83072 | 2.45719 | 1.95793 |
| concat_aligned_high | causal | False | half01 | 20.12716 | 0.83847 | 2.27640 | 1.94351 |
| concat_aligned_high | causal | False | raw | 20.12280 | 0.83869 | 2.40788 | 1.93679 |
| concat_aligned_high | causal | True | half01 | 20.12716 | 0.83847 | 2.27640 | 1.94351 |
| concat_aligned_high | causal | True | raw | 20.12280 | 0.83869 | 2.40788 | 1.93679 |
| concat_aligned_high | current | False | half01 | 20.12742 | 0.83845 | 2.28271 | 1.94373 |
| concat_aligned_high | current | False | raw | 20.12294 | 0.83865 | 2.41648 | 1.93731 |
| concat_aligned_high | current | True | half01 | 20.12742 | 0.83845 | 2.28271 | 1.94373 |
| concat_aligned_high | current | True | raw | 20.12294 | 0.83865 | 2.41648 | 1.93731 |
| concat_context | causal | False | half01 | 20.28033 | 0.84148 | 2.26186 | 1.95332 |
| concat_context | causal | False | raw | 20.27533 | 0.84171 | 2.39163 | 1.94621 |
| concat_context | causal | True | half01 | 20.28033 | 0.84148 | 2.26186 | 1.95332 |
| concat_context | causal | True | raw | 20.27533 | 0.84171 | 2.39163 | 1.94621 |
| concat_context | current | False | half01 | 20.25815 | 0.84120 | 2.26337 | 1.95461 |
| concat_context | current | False | raw | 20.25297 | 0.84142 | 2.39438 | 1.94769 |
| concat_context | current | True | half01 | 20.25815 | 0.84120 | 2.26337 | 1.95461 |
| concat_context | current | True | raw | 20.25297 | 0.84142 | 2.39438 | 1.94769 |
| concat_high | causal | False | half01 | 20.15614 | 0.83765 | 2.28784 | 1.95286 |
| concat_high | causal | False | raw | 20.15115 | 0.83778 | 2.42395 | 1.94686 |
| concat_high | causal | True | half01 | 20.15614 | 0.83765 | 2.28784 | 1.95286 |
| concat_high | causal | True | raw | 20.15115 | 0.83778 | 2.42395 | 1.94686 |
| concat_high | current | False | half01 | 20.13246 | 0.83729 | 2.28732 | 1.95487 |
| concat_high | current | False | raw | 20.12731 | 0.83741 | 2.42412 | 1.94905 |
| concat_high | current | True | half01 | 20.13246 | 0.83729 | 2.28732 | 1.95487 |
| concat_high | current | True | raw | 20.12731 | 0.83741 | 2.42412 | 1.94905 |
| copy_high | causal | False | half01 | 20.29789 | 0.83822 | 2.24969 | 1.97380 |
| copy_high | causal | False | raw | 20.29341 | 0.83848 | 2.37784 | 1.96697 |
| copy_high | causal | True | half01 | 20.29789 | 0.83822 | 2.24969 | 1.97380 |
| copy_high | causal | True | raw | 20.29341 | 0.83848 | 2.37784 | 1.96697 |
| plain_high | causal | False | half01 | 19.31045 | 0.83250 | 2.36124 | 1.96236 |
| plain_high | causal | False | raw | 19.30433 | 0.83263 | 2.50660 | 1.95704 |
| plain_high | causal | True | half01 | 19.31045 | 0.83250 | 2.36124 | 1.96236 |
| plain_high | causal | True | raw | 19.30433 | 0.83263 | 2.50660 | 1.95704 |

## dev

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_context | causal | False | half01 | 23.36276 | 0.89202 | 2.39384 | 1.97773 |
| concat_aligned_context | causal | False | raw | 23.38471 | 0.89380 | 2.45471 | 1.95531 |
| concat_aligned_context | causal | True | half01 | 23.36276 | 0.89202 | 2.39384 | 1.97773 |
| concat_aligned_context | causal | True | raw | 23.38471 | 0.89380 | 2.45471 | 1.95531 |
| concat_aligned_context | current | False | half01 | 23.36479 | 0.89209 | 2.39555 | 1.97703 |
| concat_aligned_context | current | False | raw | 23.38605 | 0.89384 | 2.45962 | 1.95512 |
| concat_aligned_context | current | True | half01 | 23.36479 | 0.89209 | 2.39555 | 1.97703 |
| concat_aligned_context | current | True | raw | 23.38605 | 0.89384 | 2.45962 | 1.95512 |
| concat_aligned_high | causal | False | half01 | 23.96467 | 0.89674 | 2.38588 | 1.95963 |
| concat_aligned_high | causal | False | raw | 23.98574 | 0.89852 | 2.44583 | 1.93736 |
| concat_aligned_high | causal | True | half01 | 23.96467 | 0.89674 | 2.38588 | 1.95963 |
| concat_aligned_high | causal | True | raw | 23.98574 | 0.89852 | 2.44583 | 1.93736 |
| concat_aligned_high | current | False | half01 | 23.96597 | 0.89678 | 2.38693 | 1.95943 |
| concat_aligned_high | current | False | raw | 23.98659 | 0.89853 | 2.44947 | 1.93756 |
| concat_aligned_high | current | True | half01 | 23.96597 | 0.89678 | 2.38693 | 1.95943 |
| concat_aligned_high | current | True | raw | 23.98659 | 0.89853 | 2.44947 | 1.93756 |
| concat_context | causal | False | half01 | 23.84184 | 0.89609 | 2.38574 | 1.96377 |
| concat_context | causal | False | raw | 23.86154 | 0.89786 | 2.44707 | 1.94136 |
| concat_context | causal | True | half01 | 23.84184 | 0.89609 | 2.38574 | 1.96377 |
| concat_context | causal | True | raw | 23.86154 | 0.89786 | 2.44707 | 1.94136 |
| concat_context | current | False | half01 | 23.87630 | 0.89632 | 2.38579 | 1.96383 |
| concat_context | current | False | raw | 23.89628 | 0.89807 | 2.44845 | 1.94167 |
| concat_context | current | True | half01 | 23.87630 | 0.89632 | 2.38579 | 1.96383 |
| concat_context | current | True | raw | 23.89628 | 0.89807 | 2.44845 | 1.94167 |
| concat_high | causal | False | half01 | 24.17708 | 0.89880 | 2.39662 | 1.95144 |
| concat_high | causal | False | raw | 24.19636 | 0.90053 | 2.46223 | 1.92928 |
| concat_high | causal | True | half01 | 24.17708 | 0.89880 | 2.39662 | 1.95144 |
| concat_high | causal | True | raw | 24.19636 | 0.90053 | 2.46223 | 1.92928 |
| concat_high | current | False | half01 | 24.20918 | 0.89897 | 2.39649 | 1.95206 |
| concat_high | current | False | raw | 24.22869 | 0.90069 | 2.46312 | 1.93013 |
| concat_high | current | True | half01 | 24.20918 | 0.89897 | 2.39649 | 1.95206 |
| concat_high | current | True | raw | 24.22869 | 0.90069 | 2.46312 | 1.93013 |
| copy_high | causal | False | half01 | 23.83171 | 0.89478 | 2.38184 | 1.98622 |
| copy_high | causal | False | raw | 23.85362 | 0.89661 | 2.44131 | 1.96360 |
| copy_high | causal | True | half01 | 23.83171 | 0.89478 | 2.38184 | 1.98622 |
| copy_high | causal | True | raw | 23.85362 | 0.89661 | 2.44131 | 1.96360 |
| plain_high | causal | False | half01 | 23.53687 | 0.89668 | 2.40121 | 1.96012 |
| plain_high | causal | False | raw | 23.55621 | 0.89835 | 2.46665 | 1.93934 |
| plain_high | causal | True | half01 | 23.53687 | 0.89668 | 2.40121 | 1.96012 |
| plain_high | causal | True | raw | 23.55621 | 0.89835 | 2.46665 | 1.93934 |

## dev_native

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_context | causal | False | half01 | 24.09473 | 0.90095 | 3.63452 | 1.60967 |
| concat_aligned_context | causal | False | raw | 24.06287 | 0.89370 | 4.03390 | 1.68534 |
| concat_aligned_context | causal | True | half01 | 24.09473 | 0.90095 | 3.63452 | 1.60967 |
| concat_aligned_context | causal | True | raw | 24.06287 | 0.89370 | 4.03390 | 1.68534 |
| concat_aligned_context | current | False | half01 | 24.09800 | 0.90069 | 3.64874 | 1.61248 |
| concat_aligned_context | current | False | raw | 24.06302 | 0.89328 | 4.05872 | 1.68985 |
| concat_aligned_context | current | True | half01 | 24.09800 | 0.90069 | 3.64874 | 1.61248 |
| concat_aligned_context | current | True | raw | 24.06302 | 0.89328 | 4.05872 | 1.68985 |
| concat_aligned_high | causal | False | half01 | 25.12055 | 0.90410 | 3.60308 | 1.60444 |
| concat_aligned_high | causal | False | raw | 25.07659 | 0.89679 | 4.00103 | 1.68166 |
| concat_aligned_high | causal | True | half01 | 25.12055 | 0.90410 | 3.60308 | 1.60444 |
| concat_aligned_high | causal | True | raw | 25.07659 | 0.89679 | 4.00103 | 1.68166 |
| concat_aligned_high | current | False | half01 | 25.12469 | 0.90393 | 3.61243 | 1.60616 |
| concat_aligned_high | current | False | raw | 25.07718 | 0.89650 | 4.01942 | 1.68466 |
| concat_aligned_high | current | True | half01 | 25.12469 | 0.90393 | 3.61243 | 1.60616 |
| concat_aligned_high | current | True | raw | 25.07718 | 0.89650 | 4.01942 | 1.68466 |
| concat_context | causal | False | half01 | 24.82394 | 0.90299 | 3.64222 | 1.59466 |
| concat_context | causal | False | raw | 24.78096 | 0.89553 | 4.05195 | 1.67273 |
| concat_context | causal | True | half01 | 24.82394 | 0.90299 | 3.64222 | 1.59466 |
| concat_context | causal | True | raw | 24.78096 | 0.89553 | 4.05195 | 1.67273 |
| concat_context | current | False | half01 | 24.86715 | 0.90332 | 3.64138 | 1.59339 |
| concat_context | current | False | raw | 24.82308 | 0.89584 | 4.05318 | 1.67165 |
| concat_context | current | True | half01 | 24.86715 | 0.90332 | 3.64138 | 1.59339 |
| concat_context | current | True | raw | 24.82308 | 0.89584 | 4.05318 | 1.67165 |
| concat_high | causal | False | half01 | 25.51777 | 0.90291 | 3.69143 | 1.60960 |
| concat_high | causal | False | raw | 25.46087 | 0.89524 | 4.10988 | 1.68899 |
| concat_high | causal | True | half01 | 25.51777 | 0.90291 | 3.69143 | 1.60960 |
| concat_high | causal | True | raw | 25.46087 | 0.89524 | 4.10988 | 1.68899 |
| concat_high | current | False | half01 | 25.56492 | 0.90325 | 3.68742 | 1.60799 |
| concat_high | current | False | raw | 25.50696 | 0.89558 | 4.10690 | 1.68744 |
| concat_high | current | True | half01 | 25.56492 | 0.90325 | 3.68742 | 1.60799 |
| concat_high | current | True | raw | 25.50696 | 0.89558 | 4.10690 | 1.68744 |
| copy_high | causal | False | half01 | 25.52364 | 0.90442 | 3.58822 | 1.59254 |
| copy_high | causal | False | raw | 25.47741 | 0.89717 | 3.98661 | 1.66825 |
| copy_high | causal | True | half01 | 25.52364 | 0.90442 | 3.58822 | 1.59254 |
| copy_high | causal | True | raw | 25.47741 | 0.89717 | 3.98661 | 1.66825 |
| plain_high | causal | False | half01 | 25.17115 | 0.90365 | 3.63861 | 1.60651 |
| plain_high | causal | False | raw | 25.11754 | 0.89606 | 4.05308 | 1.68758 |
| plain_high | causal | True | half01 | 25.17115 | 0.90365 | 3.63861 | 1.60651 |
| plain_high | causal | True | raw | 25.11754 | 0.89606 | 4.05308 | 1.68758 |

## legacy_synthetic

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_context | causal | False | half01 | 20.80277 | 0.63528 | 10.51110 | 5.04003 |
| concat_aligned_context | causal | False | raw | 20.60946 | 0.61033 | 11.81236 | 5.35759 |
| concat_aligned_context | causal | True | half01 | 20.80277 | 0.63528 | 10.51110 | 5.04003 |
| concat_aligned_context | causal | True | raw | 20.60946 | 0.61033 | 11.81236 | 5.35759 |
| concat_aligned_context | current | False | half01 | 20.79269 | 0.63420 | 10.59167 | 5.05339 |
| concat_aligned_context | current | False | raw | 20.59569 | 0.60903 | 11.91248 | 5.37498 |
| concat_aligned_context | current | True | half01 | 20.79269 | 0.63420 | 10.59167 | 5.05339 |
| concat_aligned_context | current | True | raw | 20.59569 | 0.60903 | 11.91248 | 5.37498 |
| concat_aligned_high | causal | False | half01 | 20.70019 | 0.63324 | 10.60436 | 5.06746 |
| concat_aligned_high | causal | False | raw | 20.50967 | 0.60820 | 11.92146 | 5.39253 |
| concat_aligned_high | causal | True | half01 | 20.70019 | 0.63324 | 10.60436 | 5.06746 |
| concat_aligned_high | causal | True | raw | 20.50967 | 0.60820 | 11.92146 | 5.39253 |
| concat_aligned_high | current | False | half01 | 20.69283 | 0.63252 | 10.66348 | 5.07576 |
| concat_aligned_high | current | False | raw | 20.49970 | 0.60732 | 11.99501 | 5.40325 |
| concat_aligned_high | current | True | half01 | 20.69283 | 0.63252 | 10.66348 | 5.07576 |
| concat_aligned_high | current | True | raw | 20.49970 | 0.60732 | 11.99501 | 5.40325 |
| concat_context | causal | False | half01 | 20.87854 | 0.63561 | 10.59378 | 5.05901 |
| concat_context | causal | False | raw | 20.68033 | 0.61041 | 11.91272 | 5.38456 |
| concat_context | causal | True | half01 | 20.87854 | 0.63561 | 10.59378 | 5.05901 |
| concat_context | causal | True | raw | 20.68033 | 0.61041 | 11.91272 | 5.38456 |
| concat_context | current | False | half01 | 20.86564 | 0.63507 | 10.64045 | 5.06409 |
| concat_context | current | False | raw | 20.66581 | 0.60979 | 11.96928 | 5.39084 |
| concat_context | current | True | half01 | 20.86564 | 0.63507 | 10.64045 | 5.06409 |
| concat_context | current | True | raw | 20.66581 | 0.60979 | 11.96928 | 5.39084 |
| concat_high | causal | False | half01 | 20.68477 | 0.63606 | 10.55149 | 5.03234 |
| concat_high | causal | False | raw | 20.48846 | 0.61098 | 11.86630 | 5.35280 |
| concat_high | causal | True | half01 | 20.68477 | 0.63606 | 10.55149 | 5.03234 |
| concat_high | causal | True | raw | 20.48846 | 0.61098 | 11.86630 | 5.35280 |
| concat_high | current | False | half01 | 20.67489 | 0.63565 | 10.58932 | 5.03598 |
| concat_high | current | False | raw | 20.47722 | 0.61051 | 11.91219 | 5.35730 |
| concat_high | current | True | half01 | 20.67489 | 0.63565 | 10.58932 | 5.03598 |
| concat_high | current | True | raw | 20.47722 | 0.61051 | 11.91219 | 5.35730 |
| copy_high | causal | False | half01 | 20.25592 | 0.63634 | 10.38425 | 5.01069 |
| copy_high | causal | False | raw | 20.08676 | 0.61154 | 11.67264 | 5.32312 |
| copy_high | causal | True | half01 | 20.25592 | 0.63634 | 10.38425 | 5.01069 |
| copy_high | causal | True | raw | 20.08676 | 0.61154 | 11.67264 | 5.32312 |
| plain_high | causal | False | half01 | 20.66436 | 0.63087 | 10.71417 | 5.09433 |
| plain_high | causal | False | raw | 20.47050 | 0.60558 | 12.05179 | 5.42512 |
| plain_high | causal | True | half01 | 20.66436 | 0.63087 | 10.71417 | 5.09433 |
| plain_high | causal | True | raw | 20.47050 | 0.60558 | 12.05179 | 5.42512 |

## event_cut

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_context | causal | False | half01 | 22.99052 | 0.57384 | 12.16444 | 5.22097 |
| concat_aligned_context | causal | False | raw | 22.68742 | 0.54726 | 13.72482 | 5.59736 |
| concat_aligned_context | causal | True | half01 | 23.01322 | 0.57378 | 12.14772 | 5.22446 |
| concat_aligned_context | causal | True | raw | 22.68962 | 0.54727 | 13.72289 | 5.59721 |
| concat_aligned_context | current | False | half01 | 22.98399 | 0.57271 | 12.24035 | 5.23520 |
| concat_aligned_context | current | False | raw | 22.67463 | 0.54596 | 13.82388 | 5.61560 |
| concat_aligned_context | current | True | half01 | 23.00262 | 0.57262 | 12.23051 | 5.23912 |
| concat_aligned_context | current | True | raw | 22.67463 | 0.54596 | 13.82388 | 5.61560 |
| concat_aligned_high | causal | False | half01 | 22.93890 | 0.57580 | 12.09475 | 5.20726 |
| concat_aligned_high | causal | False | raw | 22.64462 | 0.54938 | 13.64686 | 5.58260 |
| concat_aligned_high | causal | True | half01 | 22.96075 | 0.57573 | 12.07879 | 5.21056 |
| concat_aligned_high | causal | True | raw | 22.64662 | 0.54939 | 13.64482 | 5.58230 |
| concat_aligned_high | current | False | half01 | 22.93689 | 0.57517 | 12.14116 | 5.21450 |
| concat_aligned_high | current | False | raw | 22.63900 | 0.54866 | 13.70837 | 5.59201 |
| concat_aligned_high | current | True | half01 | 22.95599 | 0.57509 | 12.13059 | 5.21819 |
| concat_aligned_high | current | True | raw | 22.63900 | 0.54866 | 13.70837 | 5.59201 |
| concat_context | causal | False | half01 | 22.96723 | 0.57605 | 12.10649 | 5.20410 |
| concat_context | causal | False | raw | 22.67104 | 0.54944 | 13.66328 | 5.58107 |
| concat_context | causal | True | half01 | 22.98999 | 0.57597 | 12.08859 | 5.20708 |
| concat_context | causal | True | raw | 22.67404 | 0.54946 | 13.65865 | 5.58016 |
| concat_context | current | False | half01 | 22.93820 | 0.57548 | 12.14645 | 5.20888 |
| concat_context | current | False | raw | 22.64176 | 0.54888 | 13.71450 | 5.58673 |
| concat_context | current | True | half01 | 22.95789 | 0.57540 | 12.13620 | 5.21260 |
| concat_context | current | True | raw | 22.64176 | 0.54888 | 13.71450 | 5.58673 |
| concat_high | causal | False | half01 | 22.91790 | 0.57047 | 12.28956 | 5.25387 |
| concat_high | causal | False | raw | 22.61070 | 0.54397 | 13.87300 | 5.63395 |
| concat_high | causal | True | half01 | 22.93798 | 0.57037 | 12.27299 | 5.25707 |
| concat_high | causal | True | raw | 22.61215 | 0.54398 | 13.86862 | 5.63314 |
| concat_high | current | False | half01 | 22.89382 | 0.57007 | 12.31966 | 5.25626 |
| concat_high | current | False | raw | 22.58679 | 0.54358 | 13.91196 | 5.63689 |
| concat_high | current | True | half01 | 22.91264 | 0.56998 | 12.30986 | 5.26012 |
| concat_high | current | True | raw | 22.58679 | 0.54358 | 13.91196 | 5.63689 |
| copy_high | causal | False | half01 | 22.69577 | 0.57612 | 11.97860 | 5.19439 |
| copy_high | causal | False | raw | 22.41869 | 0.54951 | 13.52557 | 5.56588 |
| copy_high | causal | True | half01 | 22.71559 | 0.57604 | 11.96896 | 5.19800 |
| copy_high | causal | True | raw | 22.41869 | 0.54951 | 13.52557 | 5.56588 |
| plain_high | causal | False | half01 | 22.85531 | 0.57314 | 12.24364 | 5.24718 |
| plain_high | causal | False | raw | 22.55646 | 0.54661 | 13.82522 | 5.62917 |
| plain_high | causal | True | half01 | 22.87457 | 0.57305 | 12.23384 | 5.25110 |
| plain_high | causal | True | raw | 22.55646 | 0.54661 | 13.82522 | 5.62917 |

## event_light_step

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_context | causal | False | half01 | 23.69939 | 0.57073 | 12.10506 | 4.85581 |
| concat_aligned_context | causal | False | raw | 23.36054 | 0.54325 | 13.50874 | 5.21742 |
| concat_aligned_context | causal | True | half01 | 23.72466 | 0.56983 | 12.04439 | 4.86974 |
| concat_aligned_context | causal | True | raw | 23.37115 | 0.54330 | 13.47907 | 5.21672 |
| concat_aligned_context | current | False | half01 | 23.69623 | 0.56956 | 12.14008 | 4.86979 |
| concat_aligned_context | current | False | raw | 23.35099 | 0.54187 | 13.57327 | 5.23532 |
| concat_aligned_context | current | True | half01 | 23.70887 | 0.56859 | 12.12162 | 4.88452 |
| concat_aligned_context | current | True | raw | 23.35099 | 0.54187 | 13.57327 | 5.23532 |
| concat_aligned_high | causal | False | half01 | 23.89290 | 0.57669 | 11.80896 | 4.78303 |
| concat_aligned_high | causal | False | raw | 23.56747 | 0.54944 | 13.17524 | 5.13710 |
| concat_aligned_high | causal | True | half01 | 23.92785 | 0.57587 | 11.74300 | 4.79638 |
| concat_aligned_high | causal | True | raw | 23.58009 | 0.54952 | 13.14472 | 5.13610 |
| concat_aligned_high | current | False | half01 | 23.90105 | 0.57616 | 11.80836 | 4.78925 |
| concat_aligned_high | current | False | raw | 23.57126 | 0.54879 | 13.19753 | 5.14544 |
| concat_aligned_high | current | True | half01 | 23.92183 | 0.57524 | 11.78586 | 4.80367 |
| concat_aligned_high | current | True | raw | 23.57126 | 0.54879 | 13.19753 | 5.14544 |
| concat_context | causal | False | half01 | 24.00062 | 0.57644 | 11.90689 | 4.79266 |
| concat_context | causal | False | raw | 23.65701 | 0.54889 | 13.28704 | 5.14968 |
| concat_context | causal | True | half01 | 24.03183 | 0.57560 | 11.84123 | 4.80602 |
| concat_context | causal | True | raw | 23.66871 | 0.54897 | 13.25604 | 5.14855 |
| concat_context | current | False | half01 | 23.98904 | 0.57595 | 11.90690 | 4.79752 |
| concat_context | current | False | raw | 23.64357 | 0.54837 | 13.30623 | 5.15556 |
| concat_context | current | True | half01 | 24.00712 | 0.57501 | 11.88443 | 4.81201 |
| concat_context | current | True | raw | 23.64357 | 0.54837 | 13.30623 | 5.15556 |
| concat_high | causal | False | half01 | 23.73333 | 0.57032 | 12.04735 | 4.84008 |
| concat_high | causal | False | raw | 23.39620 | 0.54307 | 13.44834 | 5.20031 |
| concat_high | causal | True | half01 | 23.76110 | 0.56949 | 11.98681 | 4.85343 |
| concat_high | causal | True | raw | 23.40756 | 0.54316 | 13.41865 | 5.19899 |
| concat_high | current | False | half01 | 23.72768 | 0.57005 | 12.03833 | 4.84237 |
| concat_high | current | False | raw | 23.38891 | 0.54278 | 13.45644 | 5.20329 |
| concat_high | current | True | half01 | 23.74271 | 0.56911 | 12.01920 | 4.85703 |
| concat_high | current | True | raw | 23.38891 | 0.54278 | 13.45644 | 5.20329 |
| copy_high | causal | False | half01 | 23.82718 | 0.57551 | 11.83910 | 4.80761 |
| copy_high | causal | False | raw | 23.49781 | 0.54797 | 13.22495 | 5.16383 |
| copy_high | causal | True | half01 | 23.84697 | 0.57458 | 11.81354 | 4.82186 |
| copy_high | causal | True | raw | 23.49781 | 0.54797 | 13.22495 | 5.16383 |
| plain_high | causal | False | half01 | 23.71099 | 0.57092 | 12.09106 | 4.87108 |
| plain_high | causal | False | raw | 23.37792 | 0.54345 | 13.51084 | 5.23668 |
| plain_high | causal | True | half01 | 23.72908 | 0.56999 | 12.06815 | 4.88581 |
| plain_high | causal | True | raw | 23.37792 | 0.54345 | 13.51084 | 5.23668 |

