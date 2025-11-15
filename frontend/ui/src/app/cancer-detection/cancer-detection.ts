import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';

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
  isSubmitting: boolean = false;

  constructor(private fb: FormBuilder) {
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

  onSubmit(): void { //TODO: finish
    this.cancerForm.markAllAsTouched();

    if (this.cancerForm.invalid || !this.selectedFile) {
      return; 
    }

  }
}


