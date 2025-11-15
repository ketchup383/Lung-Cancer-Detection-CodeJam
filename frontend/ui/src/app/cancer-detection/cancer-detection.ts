import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';

interface AnalysisResponse {
  result: string;
}

@Component({
  selector: 'app-cancer-detection',
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './cancer-detection.html',
  standalone: true,
  styleUrl: './cancer-detection.css',
})

export class CancerDetection {
  cancerForm: FormGroup;
  selectedFile: File | null = null;
  
  tryAgain: boolean = true;

  analysisResult: string | null = null;
  errorMessage: string | null = null;

  constructor(private fb: FormBuilder, private http: HttpClient) {
    this.cancerForm = this.fb.group({
      medicare: ['', Validators.required],
      photo: [null, [Validators.required]]
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files ? input.files[0] : null;

    this.selectedFile = file;

    this.cancerForm.patchValue({ photo: file });
    this.cancerForm.get('photo')?.updateValueAndValidity();
  }

  determineResults(): string {
    return this.analysisResult ? this.analysisResult : '';
  }

  informationReceived(): boolean {
    return this.analysisResult !== null || this.errorMessage !== null;
  }

  onTryAgain(): void {
    this.tryAgain = true;
    this.analysisResult = null;
    this.errorMessage = null;
    this.cancerForm.reset();
    this.selectedFile = null; 
  }

  onSubmit(): void {
    this.cancerForm.markAllAsTouched();
    this.tryAgain = false;

    if (this.cancerForm.invalid || !this.selectedFile) {
      return; 
    }

    this.errorMessage = null;
    const formData = new FormData();
    formData.append('photo', this.selectedFile, this.selectedFile.name);

    this.http.post<AnalysisResponse>('/predict', formData).subscribe({
      next: (response) => {
        this.analysisResult = response.result; 
        console.log('Backend response:', response);
      },
      error: (err) => {
        this.errorMessage = 'Something went wrong analyzing the image.';
        console.error(err);
      },
    });

  }
}


