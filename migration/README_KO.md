# 2026-09-05 연구 마이그레이션 백업

이 디렉터리는 새 컴퓨터로 SpecTUS 연구 상태를 옮기기 위한 생성 도구와 환경 명세를
보관한다. 잠금 holdout, OOF prediction, grouped-CV checkpoint가 포함된 연구 ZIP은
민감한 연구 산출물이므로 공개 GitHub에 업로드하지 않는다. 두 컴퓨터 간 LAN 전송이나
접근 제한된 개인 클라우드 저장소를 사용한다.

## 필요한 파일

- `dist/spectus_research_migration_2026-09-05.zip`: 이번 연구의 코드·문서·분석·잠금
  split·OOF prediction·grouped-CV checkpoint
- `dist/spectus_r32_all_attention_offline_2026-09-02.zip`: 기본 SpecTUS 모델을 포함한
  독립 실행 패키지
- `dist/MIGRATION_SHA256SUMS`: 전송 후 검증할 최상위 checksum

새 컴퓨터에서 세 파일을 같은 `dist/` 디렉터리에 둔 뒤 다음을 실행한다.

```bash
cd /path/to/SpecTUS/dist
sha256sum -c MIGRATION_SHA256SUMS
```

모든 항목이 `OK`여야 한다. 그 뒤 연구 ZIP을 새 디렉터리에 해제하고
`migration/trainSpectus_environment_2026-09-05.yml`로 Conda 환경을 복원한다.

```bash
conda env create -f migration/trainSpectus_environment_2026-09-05.yml
```

기존 컴퓨터는 새 컴퓨터에서 checksum, 환경 생성, 패키지 검증, 1-row smoke test가 모두
통과한 뒤에만 포맷한다.

