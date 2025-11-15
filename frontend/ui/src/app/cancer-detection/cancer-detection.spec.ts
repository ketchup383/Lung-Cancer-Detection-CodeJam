import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CancerDetection } from './cancer-detection';

describe('CancerDetection', () => {
  let component: CancerDetection;
  let fixture: ComponentFixture<CancerDetection>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CancerDetection]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CancerDetection);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
