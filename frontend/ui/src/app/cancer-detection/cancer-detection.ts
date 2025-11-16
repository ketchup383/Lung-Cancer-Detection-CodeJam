import { Component, signal, inject } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http'; 
import { FormsModule } from '@angular/forms';

// --- Configuration and Interfaces ---
const API_BASE_URL = 'http://localhost:8000'; 

interface PredictionResponse {
    image_id: string; 
    result: 'Cancerous' | 'Non-Cancerous' | string;
    probability: number;
    is_cancerous: boolean;
}

@Component({
  selector: 'app-cancer-detection',
  // Includes all necessary modules for form, common directives, and HTTP requests
  imports: [ReactiveFormsModule, CommonModule, HttpClientModule, FormsModule], 
  templateUrl: './cancer-detection.html', 
  styleUrl: './cancer-detection.css',
  standalone: true,
})
export class CancerDetection {
  // --- Signals for state management ---
  selectedFile = signal<File | null>(null);
  analysisResult = signal<string | null>(null);
  confidence = signal<number>(0); 
  errorMessage = signal<string | null>(null);
  isLoading = signal<boolean>(false); 
  tryAgain: boolean = true;
  
  // Dependency Injection 
  private fb = inject(FormBuilder);
  private http = inject(HttpClient);
  
  cancerForm: FormGroup;

  constructor() {
    this.cancerForm = this.fb.group({
      medicare: ['', Validators.required],
      photo: [null, [Validators.required]]
    });
  }

  informationReceived(): boolean {
    return this.analysisResult !== null || this.errorMessage !== null;
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files ? input.files[0] : null;

    this.selectedFile.set(file);
    this.cancerForm.patchValue({ photo: file });
    this.cancerForm.get('photo')?.updateValueAndValidity();
    this.errorMessage.set(null);
  }

  onTryAgain(): void {
    this.analysisResult.set(null);
    this.errorMessage.set(null);
    this.confidence.set(0);
    this.isLoading.set(false);
    this.cancerForm.reset();
    this.selectedFile.set(null);
    this.cancerForm.markAsUntouched();
    this.tryAgain = true;
  }

  onSubmit(): void {
    this.cancerForm.markAllAsTouched();
    this.tryAgain = false;

    if (this.cancerForm.invalid || !this.selectedFile()) {
      if (!this.selectedFile()) {
        this.errorMessage.set('Please select an image file.');
      }
      return;
    }

    this.isLoading.set(true);
    this.errorMessage.set(null);

    const formData = new FormData();
    // *** CRITICAL FIX: The key must be 'file' to match the FastAPI endpoint signature. ***
    formData.append('file', this.selectedFile() as File, this.selectedFile()!.name);

    this.http.post<PredictionResponse>(`${API_BASE_URL}/predict`, formData).subscribe({
      next: (response) => {
        this.analysisResult.set(response.result);
        this.confidence.set(response.probability);
        this.isLoading.set(false);
      },
      error: (err) => {
        console.error('FastAPI Error:', err);
        // Provide user feedback about the error status
        this.errorMessage.set(`Analysis failed. Server responded with status ${err.status || 'Unknown'}. Check console for details.`);
        this.isLoading.set(false);
      },
    });
  }
}